# Adult Census Income Classifier (TFX and Apache Airflow Pipeline)

An end to end Machine Learning pipeline built using TensorFlow Extended (TFX) and Apache Airflow to predict whether an individual's income exceeds $50K per year based on census data.

***

## Overview

This repository contains a production ready MLOps pipeline orchestrated with Apache Airflow. The workflow handles data ingestion, feature preprocessing using `tf.Transform`, deep neural network training with Keras embeddings, model evaluation, and experiment tracking via TensorBoard.

### Key Features
* **Automated Data Ingestion:** Loads raw census data dynamically using TFX `ExampleGen`.
* **Feature Preprocessing (`tf.Transform`):** Applies Z score normalization to continuous numeric features and dynamic vocabulary lookups with Out of Vocabulary (OOV) buckets for categorical variables.
* **Deep Neural Network Architecture:** Built with the Keras Functional API, incorporating custom embedding layers, Batch Normalization, and Dropout to prevent overfitting.
* **Pipeline Orchestration:** Workflow steps are managed and scheduled automatically using Apache Airflow DAGs.
* **Experiment Tracking:** Real time loss, accuracy, and AUC logging through TensorBoard.

***

## Model Performance

The model uses early stopping monitored on validation AUC (`val_auc`). Final results on the validation split:

* **Validation AUC:** 91.65%
* **Validation Accuracy:** 85.51%
* **Validation Loss:** 0.3097
* **Precision:** ~69.68%
* **Recall:** ~70.92%

***

## Model Architecture

* **Numeric Inputs:** `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` (Scaled via Z score)
* **Categorical Inputs:** `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` (Encoded via `Embedding(input_dim=vocab_size + 2)`)
* **Hidden Dense Layers:** 128 units -> 64 units -> 32 units (ReLU activation)
* **Regularization:** Batch Normalization and Dropout (rates: 0.3, 0.2, 0.1)
* **Output:** Single binary node with Sigmoid activation (`>50K` vs `<=50K`)

***

## Environment Compatibility Note

The python bytecode in `__pycache__` targets `cpython-37`. The versions specified in `requirements.txt` (TFX 1.9.1 / Airflow 2.3.4 / TF 2.9.3) represent the last mutually compatible set on Python 3.7. **Use a Python 3.7 virtual environment** — newer TFX releases drop Airflow orchestration support, and newer Airflow releases drop Python 3.7 support.

***

## Project Structure

```text
adult_census_tfx/
├── adult_pipeline_definition.py   # create_pipeline() shared by both runners
├── adult_pipeline_airflow.py      # Airflow DAG entry point
├── adult_pipeline_local.py        # LocalDagRunner for rapid iteration without Airflow
├── adult_trainer_module.py        # preprocessing_fn + run_fn (Transform/Trainer)
├── requirements.txt               # Dependency specifications
├── eda_analysis.ipynb             # Dataset correlation and EDA notebook
├── tfma_analysis.ipynb            # TFMA fairness analysis and evaluation scaffold
└── data/
    └── adult.csv                  # Preprocessed census dataset
```

***

## Setup and Execution Guide

### Step 1: Set up the Environment

```bash
python3.7 -m venv tfx-airflow-env
source tfx-airflow-env/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Airflow Workspace

```bash
export AIRFLOW_HOME=~/airflow
airflow db init

mkdir -p $AIRFLOW_HOME/dags
ln -s $(pwd) $AIRFLOW_HOME/dags/adult_census_tfx
```

### Step 3: Initialize Airflow Admin and Start Services

```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Team \
  --lastname COMP315 \
  --role Admin \
  --email admin@example.com

airflow webserver -p 8080 &
airflow scheduler &
```

### Step 4: Execute Pipeline

1. Access `http://localhost:8080` in your web browser.
2. Locate the `adult_census_tfx` DAG, toggle the unpause switch, and trigger a run.
3. Execution outputs and metadata DB will automatically persist under `~/COMP315/airflow_pipeline_outputs/`.

### Step 5: Post Run Notebook Evaluation

* **`eda_analysis.ipynb`**: Standalone dataset analysis covering feature correlations, missing values, and baseline distributions.
* **`tfma_analysis.ipynb`**: Evaluates model slices and fairness metrics directly from MLMD outputs. Update the `EVAL_RESULT_PATH` variable with your local artifact URI generated during Step 4 before running this notebook.

### Step 6: Track Experiments in TensorBoard

```bash
tensorboard --logdir ~/COMP315/airflow_pipeline_outputs/
```
