from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType

def csv_to_parquet(
        spark,
        input_path: str,
        output_path: str,
        sample_path: str | None = None,
        infer_schema: bool = True,
        sample_frac: float = 0.10,
        label_col: str = "fraud_bool"
):
    """Convert CSV → Parquet, optionally writing a stratified sample."""
    (reader := spark.read
               .format("csv")
               .option("header", "true")
               .option("inferSchema", str(infer_schema).lower())
               .option("delimiter", ","))

    df = reader.load(input_path)
    print(f"Loaded {df.count():,} rows; schema:")
    df.printSchema()

    # Optional 10 % stratified sample (only if sample_path supplied)
    if sample_path:
        if label_col not in df.columns:
            raise ValueError(f"{label_col} column missing — cannot stratify")
        keys = [row[label_col] for row in df.select(label_col).distinct().collect()]
        fractions = {k: sample_frac for k in keys}
        df_sample = df.sampleBy(label_col, fractions, seed=42)   # balanced classes
        (df_sample
         .write
         .mode("overwrite")
         .parquet(sample_path))
        print(f"Sample ({sample_frac:.0%}) saved to {sample_path}")

    df.write.mode("overwrite").parquet(output_path)
    print(f"Full dataset written to {output_path}")

if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("Csv_Parquet_Sample")
             .getOrCreate())

    csv_to_parquet(
        spark,
        input_path="hdfs:///user/hs5413_nyu_edu/Base.csv",
        output_path="hdfs:///user/hs5413_nyu_edu/bank_fraud_data/parquet/",
        sample_path="hdfs:///user/hs5413_nyu_edu/bank_fraud_data/sample_10pct/",
        infer_schema=True,
        sample_frac=0.10
    )
    spark.stop()