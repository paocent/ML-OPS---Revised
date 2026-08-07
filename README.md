# Adult Census Income Classifier (TFX & Apache Airflow Pipeline)

An end-to-end Machine Learning pipeline built using TensorFlow Extended (TFX) and Apache Airflow to predict whether an individual's income exceeds $50K per year based on census data.

---

## Overview

This repository contains a production ready MLOps pipeline orchestrated with Apache Airflow. The workflow handles data ingestion, feature preprocessing using `tf.Transform`, deep neural network training with Keras embeddings, model evaluation, and experiment tracking via TensorBoard.

### Key Features
* **Automated Data Ingestion:** Loads raw census data dynamically using TFX `ExampleGen`.
* **Feature Preprocessing (`tf.Transform`):** Applies Z-score normalization to continuous numeric features and dynamic vocabulary lookups with Out-Of-Vocabulary (OOV) buckets for categorical variables.
* **Deep Neural Network Architecture:** Built with the Keras Functional API, incorporating custom embedding layers, Batch Normalization, and Dropout to prevent overfitting.
* **Pipeline Orchestration:** Workflow steps are managed and scheduled automatically using Apache Airflow DAGs.
* **Experiment Tracking:** Real-time loss, accuracy, and AUC logging through TensorBoard.

---

## Model Performance

The model uses early stopping monitored on validation AUC (`val_auc`). Final results on the validation split:

* **Validation AUC:** 91.65%
* **Validation Accuracy:** 85.51%
* **Validation Loss:** 0.3097
* **Precision:** ~69.68%
* **Recall:** ~70.92%

---

## Model Architecture

* **Numeric Inputs:** `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` (Scaled via Z-score)
* **Categorical Inputs:** `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` (Encoded via `Embedding(input_dim=vocab_size + 2)`)
* **Hidden Dense Layers:** 128 units -> 64 units -> 32 units (ReLU activation)
* **Regularization:** Batch Normalization and Dropout (rates: 0.3, 0.2, 0.1)
* **Output:** Single binary node with Sigmoid activation (`>50K` vs `<=50K`)

---

## Project Structure

```text
├── dags/
│   └── adult_pipeline_airflow.py   # Airflow DAG workflow definition
├── modules/
│   └── adult_trainer_module.py     # Preprocessing fn & Keras model logic
├── README.md                       # Project documentation
└── requirements.txt                # Python environment dependencies
