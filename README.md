# Adult Census Income Classifier (TFX & Apache Airflow Pipeline)

An end to end Machine Learning pipeline built using TensorFlow Extended (TFX) and Apache Airflow to predict whether an individual's income exceeds $50K per year based on census data.

***

## Overview

This repository contains a production ready MLOps pipeline orchestrated with Apache Airflow. The workflow handles data ingestion, feature preprocessing using `tf.Transform`, deep neural network training with Keras embeddings, model evaluation, and experiment tracking via TensorBoard.

### Key Features
* **Automated Data Ingestion:** Loads raw census data dynamically using TFX `ExampleGen`.
* **Feature Preprocessing (`tf.Transform`):** Applies Z score normalization to continuous numeric features and dynamic vocabulary lookups with Out Of Vocabulary (OOV) buckets for categorical variables.
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

## Project Structure

```text
dags/
  adult_pipeline_airflow.py   # Airflow DAG workflow definition
modules/
  adult_trainer_module.py     # Preprocessing fn & Keras model logic
README.md                       # Project documentation
requirements.txt                # Python environment dependencies
```

***

## Setup and Execution Guide

### Step 1: Clone Repository and Setup Files

```bash
git clone [https://github.com/paocent/ML-OPS---Revised.git](https://github.com/paocent/ML-OPS---Revised.git) && cd ML-OPS---Revised && pip install -r requirements.txt && mkdir -p ~/airflow/dags/adult_census_tfx && cp dags/adult_pipeline_airflow.py ~/airflow/dags/adult_census_tfx/ && cp modules/adult_trainer_module.py ~/airflow/dags/adult_census_tfx/
```

### Step 2: Start Airflow Services

Open Terminal 1:
```bash
airflow webserver --port 8085
```

Open Terminal 2:
```bash
airflow scheduler
```

### Step 3: Run Pipeline in UI

1. Open http://localhost:8085 in your browser.
2. Unpause the `adult_census_tfx` DAG.
3. Click the Trigger button to start execution.

### Step 4: Track Experiments in TensorBoard

```bash
tensorboard --logdir ~/COMP315/airflow_pipeline_outputs/
```
