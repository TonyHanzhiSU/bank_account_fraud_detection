# train_xgboost_adasyn.py
"""
Train an XGBoost classifier on the processed fraud‑detection data and **save it with Python’s built‑in `pickle`**.
The script performs:
  1. Load Parquet containing `[fraud_bool, features]` (Vector column).
  2. Convert the Spark Vector to a NumPy array.
  3. Oversample with ADASYN to balance classes.
  4. Fit an XGBoost classifier (n_estimators=100, learning_rate=0.1, max_depth=3).
  5. Evaluate on the original (un‑resampled) data (PR‑AUC, ROC‑AUC, report).
  6. **Persist the model using `pickle`** for later inference.

If the dataset is too large for driver memory, consider using XGBoost4J‑Spark or
`scale_pos_weight` instead of ADASYN.
"""

import argparse
import pickle

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier

from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array


def spark_to_numpy(df, label_col: str):
    """Convert a Spark DataFrame with vector features into (X, y) NumPy arrays."""
    array_df = df.select(vector_to_array("features").alias("features"), label_col)
    pdf = array_df.toPandas()
    X = np.vstack(pdf["features"].values)
    y = pdf[label_col].values.astype(int)
    return X, y


def main(processed_path: str, model_out: str, sampling_strategy: float):
    spark = SparkSession.builder.appName("Train-XGBoost-ADASYN").getOrCreate()

    print(f"Loading processed data from: {processed_path}")
    df = spark.read.parquet(processed_path)
    print(f"Loaded {df.count():,} rows.")

    # Convert to NumPy for scikit‑learn / imbalanced‑learn
    X, y = spark_to_numpy(df, label_col="fraud_bool")
    print(f"Class distribution before ADASYN: {np.bincount(y)} (neg, pos)")

    # --- ADASYN oversampling ---
    ada = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_neighbors=5)
    X_res, y_res = ada.fit_resample(X, y)
    print(f"Class distribution after ADASYN: {np.bincount(y_res)} (neg, pos)")

    # --- XGBoost classifier ---
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    clf.fit(X_res, y_res)

    # --- Evaluation on original (un‑resampled) data ---
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    print("\nClassification Report (on original data):")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"ROC‑AUC:  {roc_auc_score(y, y_prob):.4f}")
    print(f"PR‑AUC:   {average_precision_score(y, y_prob):.4f}")

    # --- Persist the model using pickle ---
    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved (pickle) to: {model_out}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with ADASYN oversampling and save via pickle.")
    parser.add_argument("--input", required=True, help="HDFS or local path to processed Parquet directory")
    parser.add_argument("--model-out", required=True, help="Destination path for the pickled model (e.g., model.pkl)")
    parser.add_argument("--sampling-strategy", type=float, default=0.5,
                        help="ADASYN sampling_strategy (e.g., 0.5 ⇒ minority will be 50% of majority)")
    args = parser.parse_args()

    main(args.input, args.model_out, args.sampling_strategy)
