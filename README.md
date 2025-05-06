# bank_account_fraud_detection
## Final Project For Big Data

## Project Overview

This project implements a big data pipeline to detect fraudulent bank account applications using the Bank Account Fraud (BAF) Tabular Dataset Suite (NeurIPS 2022). It leverages Apache Spark for scalable data processing and feature engineering, Hadoop (HDFS) for distributed storage, and XGBoost with ADASYN oversampling for machine learning model training. The project was developed and tested on the NYU Google Cloud Dataproc environment.

## Dataset

* **Name:** Bank Account Fraud (BAF) Tabular Dataset Suite
* **Source:** [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data)
* **Format:** Initially CSV, converted to Parquet for efficient processing.

## Architecture

The project follows a hybrid approach:

1.  **Batch Processing (Spark):**
    * Data ingestion from GCS/HDFS.
    * Conversion from CSV to Parquet using Spark.
    * Feature engineering using a SparkML Pipeline (One-Hot Encoding, Scaling).
2.  **Local Model Training (Python/XGBoost on Driver Node):**
    * Processed data is loaded via Spark but collected to the driver node (`.toPandas()`).
    * ADASYN oversampling is applied using `imbalanced-learn`.
    * An XGBoost model is trained using `xgboost`.
    * The model is saved locally using `pickle`.
3.  **Storage:**
    * HDFS: Stores intermediate Parquet data.
    * GCS: Used for staging raw data, scripts, and potentially the final model.

*(Note: A fully distributed training approach using SparkML's native algorithms or XGBoost4J-Spark was considered but not implemented due to environment constraints. The current implementation trains the ML model locally on the Spark driver/master node).*

## Scripts

This repository contains the following key scripts:

1.  **`csv_2_parquet.py`**
    * **Purpose:** Reads raw CSV data from HDFS or GCS and converts it into the more efficient Parquet format, saving it back to HDFS.
    * **Usage:**
        ```bash
        # Make sure HDFS_CSV_INPUT_PATH and HDFS_PARQUET_OUTPUT_PATH are set correctly inside the script
        spark-submit --master yarn --deploy-mode client csv_2_parquet.py
        ```

2.  **`feature_engineering.py`**
    * **Purpose:** Reads the Parquet data, applies a SparkML Pipeline for feature engineering (One-Hot Encoding for categoricals, Min-Max Scaling for numericals), and saves the processed data (features vector + label) back to HDFS in Parquet format.
    * **Usage:**
        ```bash
        # Make sure HDFS_PARQUET_INPUT_PATH, HDFS_PROCESSED_OUTPUT_PATH,
        # COLUMNS_TO_EXCLUDE, and LABEL_COLUMN are set correctly inside the script
        spark-submit --master yarn --deploy-mode client feature_engineering.py
        ```

3.  **`install.sh`** (Example Content - Adapt as needed)
    * **Purpose:** Installs necessary Python libraries directly onto the Dataproc master node using `pip`. This is required because the `train_xgboost_adasyn.py` script runs locally on the master node after collecting data.
    * **Example Content:**
        ```bash
        #!/bin/bash
        set -euxo pipefail
        echo "Installing Python dependencies for local training..."
        pip3 install --user --upgrade pip
        pip3 install --user numpy pandas scikit-learn imbalanced-learn xgboost
        echo "Dependencies installed."
        ```
    * **Usage:** Run directly on the Dataproc master node's terminal *before* running the training script:
        ```bash
        bash install.sh
        ```

4.  **`train_xgboost_adasyn.py`**
    * **Purpose:** Loads the processed feature data using Spark, collects it to the driver node, applies ADASYN oversampling, trains an XGBoost classifier locally, evaluates it, and saves the trained model as a pickle file to the master node's local filesystem.
    * **Prerequisites:** Run `install.sh` first to install dependencies locally on the master node.
    * **Usage:**
        ```bash
        # Run using spark-submit in client mode
        spark-submit \
          --master yarn \
          --deploy-mode client \
          train_xgboost_adasyn.py \
            --input <HDFS_or_GCS_path_to_processed_features> \
            --model-out <local_path_on_master_node_for_pkl_model> \
            --sampling-strategy <e.g., 0.5>
        ```
        *Example:*
        ```bash
        spark-submit \
          --master yarn \
          --deploy-mode client \
          train_xgboost_adasyn.py \
            --input hdfs:///user/hs5413_nyu_edu/bank_fraud_data/processed_features/ \
            --model-out ./xgb_adasyn_model.pkl \
            --sampling-strategy 0.8
        ```

## Execution Workflow

1.  **Upload Data:** Upload the raw BAF dataset CSV files to HDFS or GCS.
2.  **Convert to Parquet:** Run `csv_2_parquet.py` via `spark-submit`.
3.  **Feature Engineering:** Run `feature_engineering.py` via `spark-submit`.
4.  **Install Dependencies:** SSH into the Dataproc master node and run `bash install.sh`.
5.  **Train Model:** Run `train_xgboost_adasyn.py` via `spark-submit` (in client mode) on the master node, providing the path to the processed features and a local path for the output model.
6.  **(Optional) Upload Model:** Copy the generated `.pkl` model file from the master node's local filesystem to GCS or HDFS if needed.
