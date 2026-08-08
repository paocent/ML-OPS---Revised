# Adult Census Income Classifier

**TFX + Apache Airflow + TensorFlow Serving**

An end-to-end Machine Learning pipeline built using **TensorFlow Extended (TFX)** and **Apache Airflow** to predict whether an individual's income exceeds **$50K per year** based on census data.

---

## 📌 Overview

This repository contains a production-ready MLOps pipeline orchestrated with **Apache Airflow**.

The workflow handles:

* Data ingestion
* Feature preprocessing using `tf.Transform`
* Deep neural network training with Keras embeddings
* Model evaluation
* Model serving with TensorFlow Serving
* Experiment tracking with TensorBoard

---

## ✨ Key Features

* **Automated Data Ingestion**
  Loads raw census data dynamically using TFX `ExampleGen`.

* **Feature Preprocessing (`tf.Transform`)**
  Applies Z-score normalization to continuous numeric features and dynamic vocabulary lookups with Out-of-Vocabulary (OOV) buckets for categorical variables.

* **Deep Neural Network Architecture**
  Built using the Keras Functional API with custom embedding layers, Batch Normalization, and Dropout to reduce overfitting.

* **Pipeline Orchestration**
  Workflow steps are automatically managed and scheduled using Apache Airflow DAGs.

* **Experiment Tracking**
  Tracks loss, accuracy, and AUC metrics in real time using TensorBoard.

* **Model Serving**
  Exports the trained model for deployment using TensorFlow Serving.

---

## 📊 Model Performance

The model uses **early stopping monitored on validation AUC (`val_auc`)**.

Final results on the validation split:

| Metric                  |      Result |
| ----------------------- | ----------: |
| **Validation AUC**      |  **91.65%** |
| **Validation Accuracy** |  **85.51%** |
| **Validation Loss**     |  **0.3097** |
| **Precision**           | **~69.68%** |
| **Recall**              | **~70.92%** |

---

## 🧠 Model Architecture

### Numeric Features

The following numerical features are normalized using **Z-score normalization**:

* `age`
* `education_num`
* `capital_gain`
* `capital_loss`
* `hours_per_week`

### Categorical Features

The following categorical features are encoded using embedding layers:

* `workclass`
* `marital_status`
* `occupation`
* `relationship`
* `race`
* `sex`
* `native_country`

Embedding layers use:

```text
input_dim = vocab_size + 2
```

The additional vocabulary capacity accounts for special/OOV values.

### Neural Network

```text
Numeric Inputs ───────────────┐
                              │
Categorical Inputs → Embeddings
                              │
                              ▼
                     Concatenated Features
                              │
                              ▼
                     Dense Layer (128)
                              │
                     Batch Normalization
                              │
                         Dropout (0.3)
                              │
                              ▼
                     Dense Layer (64)
                              │
                     Batch Normalization
                              │
                         Dropout (0.2)
                              │
                              ▼
                     Dense Layer (32)
                              │
                         Dropout (0.1)
                              │
                              ▼
                     Sigmoid Output (1)
                              │
                              ▼
                    >50K / <=50K Income
```

### Regularization

* **Batch Normalization**
* **Dropout**

  * `0.3`
  * `0.2`
  * `0.1`

### Output

The model produces a single binary output using a **Sigmoid activation**:

```text
>50K
<=50K
```

---

## 🐍 Environment Compatibility

> **Important:** The Python bytecode in `__pycache__` targets **CPython 3.7**.

The versions specified in `requirements.txt` represent the last mutually compatible configuration for this project:

```text
TFX       1.9.1
Airflow   2.3.4
TensorFlow 2.9.3
Python    3.7
```

### Recommended Environment

Use a **Python 3.7 virtual environment**.

Newer TFX releases may no longer support the same Airflow orchestration setup, while newer Airflow releases have dropped Python 3.7 support.

---

## 📁 Project Structure

```text
adult_census_tfx/
│
├── adult_pipeline_definition.py
│   └── create_pipeline() shared by both runners
│
├── adult_pipeline_airflow.py
│   └── Airflow DAG entry point
│
├── adult_pipeline_local.py
│   └── LocalDagRunner for rapid iteration without Airflow
│
├── adult_trainer_module.py
│   └── preprocessing_fn + run_fn (Transform/Trainer)
│
├── requirements.txt
│   └── Dependency specifications
│
├── eda_analysis.ipynb
│   └── Dataset correlation and EDA notebook
│
├── tfma_analysis.ipynb
│   └── TFMA fairness analysis and evaluation scaffold
│
└── data/
    └── adult.csv
        └── Preprocessed census dataset
```

---

# 🚀 Setup and Execution Guide

## Step 1: Set Up the Virtual Environment

If you already created a virtual environment such as `tfx-env` or `tfx-airflow-env`, activate it:

```bash
source tfx-env/bin/activate
```

If you have not created the environment yet:

```bash
python3.7 -m venv tfx-env
source tfx-env/bin/activate
pip install -r requirements.txt
```

Verify the Python version:

```bash
python --version
```

Expected:

```text
Python 3.7.x
```

---

## Step 2: Configure Airflow Workspace

Set the Airflow home directory:

```bash
export AIRFLOW_HOME=~/airflow
```

Initialize the Airflow database:

```bash
airflow db init
```

Create the DAG directory:

```bash
mkdir -p $AIRFLOW_HOME/dags
```

Create a symbolic link to the repository:

```bash
ln -s $(pwd) $AIRFLOW_HOME/dags/adult_census_tfx
```

---

## Step 3: Initialize Airflow

Create an Airflow administrator account:

```bash
airflow users create \
    --username admin \
    --password admin \
    --firstname Team \
    --lastname COMP315 \
    --role Admin \
    --email admin@example.com
```

Start the Airflow webserver:

```bash
airflow webserver -p 8080 &
```

Start the Airflow scheduler:

```bash
airflow scheduler &
```

---

## Step 4: Execute the Pipeline

Open the Airflow web interface:

```text
http://localhost:8080
```

Then:

1. Log in using the Airflow administrator account.
2. Locate the `adult_census_tfx` DAG.
3. Toggle the DAG to **Unpaused**.
4. Trigger a new DAG run.
5. Monitor the pipeline components from the Airflow interface.

Pipeline execution outputs and the metadata database will be persisted under:

```text
~/COMP315/airflow_pipeline_outputs/
```

---

## Step 5: Containerized Model Deployment

### Build the Docker Image

Build the project Docker image:

```bash
docker build -t adult-census-tfx:latest .
```

### Run TensorFlow Serving

Serve the exported model using TensorFlow Serving:

```bash
docker run -d \
    -p 8501:8501 \
    -v ~/COMP315/airflow_pipeline_outputs/adult_census_tfx_output/Pusher/model/latest:/models/adult_census \
    -e MODEL_NAME=adult_census \
    tensorflow/serving:latest
```

### Verify the Model Server

Check that the model is available:

```bash
curl http://localhost:8501/v1/models/adult_census
```

A successful response should contain information about the loaded `adult_census` model.

---

## Step 6: Post-Run Notebook Evaluation

### `eda_analysis.ipynb`

The EDA notebook provides standalone dataset analysis, including:

* Feature correlations
* Missing-value analysis
* Baseline feature distributions
* Dataset exploration

### `tfma_analysis.ipynb`

The TFMA notebook evaluates:

* Model slices
* Evaluation metrics
* Fairness metrics
* Model performance using MLMD-generated artifacts

Before running the notebook, update:

```python
EVAL_RESULT_PATH
```

with the local artifact URI generated during the Airflow pipeline execution.

---

## Step 7: Track Experiments with TensorBoard

Launch TensorBoard using the pipeline execution outputs:

```bash
tensorboard --logdir ~/COMP315/airflow_pipeline_outputs/
```

Open TensorBoard in your browser:

```text
http://localhost:6006
```

You can monitor metrics such as:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* AUC
* Validation AUC

---

## 🛠️ Technology Stack

| Technology                          | Purpose                                |
| ----------------------------------- | -------------------------------------- |
| **Python 3.7**                      | Development environment                |
| **TensorFlow 2.9.3**                | Deep learning framework                |
| **TensorFlow Extended (TFX) 1.9.1** | ML pipeline framework                  |
| **TensorFlow Transform**            | Feature preprocessing                  |
| **Apache Airflow 2.3.4**            | Pipeline orchestration                 |
| **TFMA**                            | Model evaluation and fairness analysis |
| **Keras**                           | Neural network architecture            |
| **TensorBoard**                     | Experiment tracking                    |
| **Docker**                          | Containerization                       |
| **TensorFlow Serving**              | Model deployment                       |

---

## 🔄 Pipeline Workflow

```text
                    ┌─────────────────┐
                    │   Census Data   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    ExampleGen   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     tf.Transform│
                    │   Preprocessing  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Trainer     │
                    │  Keras + TF     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      TFMA       │
                    │ Model Evaluation│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Pusher      │
                    │  Model Export   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ TensorFlow      │
                    │    Serving      │
                    └─────────────────┘
```

---

## 📈 Results

The trained model achieves a **91.65% validation AUC** and **85.51% validation accuracy**, demonstrating strong performance in distinguishing individuals earning more than $50K annually from those earning $50K or less.

The project also demonstrates an end-to-end MLOps workflow, from raw data ingestion and preprocessing through model training, evaluation, orchestration, experiment tracking, and deployment.

---

## 👤 Author

**Paolo Adame**

Adult Census Income Classification — TFX & Apache Airflow Pipeline
