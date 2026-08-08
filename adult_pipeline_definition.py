"""
Shared TFX pipeline definition for the Adult Census Income dataset.

This module only builds the `tfx.dsl.Pipeline` object (Step 10 of the
assignment) — it has no orchestrator-specific code, so it can be imported
by both:
  - adult_pipeline_airflow.py  (builds the DAG for Airflow to discover)
  - adult_pipeline_local.py    (a single script you can just `python` run)
"""
import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.orchestration import data_types, metadata, pipeline
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# ─────────────────────────────────────────────
# Pipeline constants
# ─────────────────────────────────────────────
PIPELINE_NAME = "adult_census_tfx"
BASE_DIR = os.path.abspath(
    os.path.join(os.path.expanduser("~"), "COMP315")
)
PROJECT_ROOT = os.path.join(BASE_DIR, "adult_census_tfx")

# Root directory for all pipeline artifacts
PIPELINE_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "airflow_pipeline_outputs", "adult_census_tfx_output")
)

# Where raw CSV data lives
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# Module containing preprocessing and trainer logic.
MODULE_FILE = os.path.join(PROJECT_ROOT, "adult_trainer_module.py")

# SQLite metadata store
METADATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "airflow_pipeline_outputs", "metadata", PIPELINE_NAME, "metadata.db")
)

# Where the final SavedModel is pushed
SERVING_MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "airflow_pipeline_outputs", "tfx_output", "serving_model", PIPELINE_NAME)
)

# Training hyperparameters
TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 200

LABEL_KEY = "target_xf"
ACCURACY_THRESHOLD = 0.75


# ─────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────
def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    metadata_path: str,
    serving_model_dir: str,
    train_num_steps: int,
    eval_num_steps: int,
    accuracy_threshold: float = ACCURACY_THRESHOLD,
    enable_cache: bool = True,
    use_runtime_data_root_param: bool = True,
) -> pipeline.Pipeline:
    """Builds the Adult Census TFX pipeline.

    use_runtime_data_root_param: Airflow can override `data_root` at
    trigger time via a RuntimeParameter, which is handy for scheduled
    orchestration. The local runner has no such trigger-time UI, so it
    passes a plain string instead — set this to False in that case.
    """

    if use_runtime_data_root_param:
        example_gen_input = data_types.RuntimeParameter(
            "data_root",
            ptype=str,
            default=data_root,
        )
    else:
        example_gen_input = data_root

    # 1. ExampleGen: ingest the CSV, split 2:1 train/eval, convert to
    #    TFRecord. (URIs of the resulting splits are printed by whichever
    #    runner script calls this function — see adult_pipeline_local.py.)
    example_gen = CsvExampleGen(input_base=example_gen_input)

    # 2. StatisticsGen: per-feature stats (min/max/mean/missing/histograms)
    #    for both the train and eval splits.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # 3. SchemaGen: infer types, ranges, and categorical vocab from stats.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # 4. ExampleValidator: compare stats against schema, flag anomalies.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Transform: feature engineering via preprocessing_fn in MODULE_FILE
    #    (drops "fnlwgt" and "education" — see adult_trainer_module.py).
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=MODULE_FILE,
    )

    # 6. Trainer: builds and fits the Keras model via run_fn in MODULE_FILE.
    trainer = Trainer(
        module_file=MODULE_FILE,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=train_num_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_num_steps),
    )

    # 7. Resolver: find the latest blessed model to use as the Evaluator's
    #    baseline. On the very first run no blessed model exists yet, so
    #    the Evaluator automatically blesses the first model.
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # 8. Evaluator: TFMA evaluation, sliced overall + by sex_xf/race_xf,
    #    must beat the resolved baseline's accuracy threshold to be blessed.
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key=LABEL_KEY,
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["sex_xf"]),
            tfma.SlicingSpec(feature_keys=["race_xf"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": accuracy_threshold}
                            )
                        ),
                    ),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # 9. Pusher: only copies the model to the serving dir if blessed.
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
    )
