import tensorflow as tf
import tensorflow_transform as tft

# ─────────────────────────────────────────────
# Hyperparameters and Feature definitions
# ─────────────────────────────────────────────
EPOCHS = 10
LEARNING_RATE = 1e-3 #to convert this to decimal 0.001

#Default Rate: 0.001 (1e-3)

# 0.0005 (5e-4)

# 0.0001 (1e-4)

# 0.00001 (1e-5)

# ─────────────────────────────────────────────
# Features dropped from the model (present in the raw CSV, intentionally
# left out of preprocessing_fn's `outputs`, so they never reach Transform's
# output / the Trainer):
#
#   - "fnlwgt": the Census Bureau's sampling weight — how many people in
#     the population a given row is meant to represent. It reflects survey
#     methodology, not anything about the person's income, so including it
#     would let the model latch onto a spurious, non-causal signal.
#
#   - "education": a categorical restatement of "education-num" (e.g.
#     "Bachelors" vs 13). We keep the numeric, ordinal version and drop the
#     string duplicate to avoid feeding the same information into the model
#     twice under two different encodings.
#
# Everything else is kept, including "relationship" and "marital-status"
# even though they overlap somewhat (household role vs. civil status) —
# each carries information the other doesn't.
# ─────────────────────────────────────────────

# Continuous numeric features matching dataset schema
NUMERIC_FEATURES = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

# Categorical features matching dataset schema
CATEGORICAL_FEATURES = {
    "workclass": 10,
    "marital-status": 7,
    "occupation": 15,
    "relationship": 6,
    "race": 5,
    "sex": 2,
    "native-country": 42,
}

LABEL_KEY = "target"


def _transformed_name(key: str) -> str:
    return key + "_xf"


# ─────────────────────────────────────────────
# Preprocessing function (used by Transform component)
# ─────────────────────────────────────────────
def preprocessing_fn(inputs):
    """tf.transform preprocessing function for Adult Census dataset.

    `inputs` contains every raw column from the schema, including "fnlwgt"
    and "education" — but we only read from NUMERIC_FEATURES and
    CATEGORICAL_FEATURES above, and only ever *write* to `outputs` for
    those. Any raw column we never write to `outputs` (fnlwgt, education)
    simply doesn't appear in the transformed TFRecords Transform produces,
    which is how those two features get dropped from the pipeline.
    """
    outputs = {}

    # Z-score normalize numeric features
    for feat in NUMERIC_FEATURES:
        outputs[_transformed_name(feat)] = tft.scale_to_z_score(
            tf.cast(inputs[feat], tf.float32)
        )

    # Compute vocabulary + int-encode categorical features
    for feat, vocab_size in CATEGORICAL_FEATURES.items():
        outputs[_transformed_name(feat)] = tft.compute_and_apply_vocabulary(
            inputs[feat],
            top_k=vocab_size,
            num_oov_buckets=1,
        )

    # Encode label column ('>50K' vs '<=50K') to integer 0/1
    outputs[_transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY],
        top_k=2
    )

    return outputs


# ─────────────────────────────────────────────
# Model builder (used by Trainer component)
# ─────────────────────────────────────────────
def _build_keras_model() -> tf.keras.Model:
    """Build a DNN for binary income classification."""
    inputs = []
    encoded = []

    # Numeric inputs
    for feat in NUMERIC_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=_transformed_name(feat))
        inputs.append(inp)
        encoded.append(inp)

    # Categorical inputs via Embeddings
    # Categorical inputs via Embeddings
    for feat, vocab_size in CATEGORICAL_FEATURES.items():
        inp = tf.keras.Input(shape=(1,), name=_transformed_name(feat), dtype=tf.int64)
        inputs.append(inp)
        emb_dim = max(1, vocab_size // 2)
        # Adding + 2 ensures room for 0-indexed vocabulary + OOV bucket index
        emb = tf.keras.layers.Embedding(input_dim=vocab_size + 2, output_dim=emb_dim)(inp)
        emb = tf.keras.layers.Flatten()(emb)
        encoded.append(emb)

    x = tf.keras.layers.concatenate(encoded)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and runs inference."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed = model.tft_layer(parsed)
        outputs = model(transformed)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a signature that applies tf.Transform to raw examples only
    (no model inference). Required by TFMA when EvalConfig's ModelSpec sets
    preprocessing_function_names=["transform_features"], since the Evaluator
    receives raw (untransformed) examples from ExampleGen and must transform
    them the same way training data was transformed before scoring."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function
    def transform_features_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern, tf_transform_output, batch_size=32, num_epochs=None):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=_transformed_name(LABEL_KEY),
        num_epochs=num_epochs
    )
    return dataset


# ─────────────────────────────────────────────
# Execution entrypoint
# ─────────────────────────────────────────────
def run_fn(fn_args):
    """TFX Trainer execution entry point."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=32, num_epochs=EPOCHS)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=32, num_epochs=1)

    log_dir = fn_args.model_run_dir

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    model = _build_keras_model()

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=3, restore_best_weights=True
            )
        ],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
        "transform_features": _get_transform_features_signature(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)