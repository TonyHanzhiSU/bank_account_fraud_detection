# Import necessary SparkML and PySpark libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    MinMaxScaler,
    StandardScaler # StandardScaler is often preferred over MinMaxScaler, included as an alternative
)
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, DoubleType, IntegerType, FloatType, LongType, ShortType, ByteType

def create_feature_engineering_pipeline(df, input_cols_to_exclude=None, label_col="fraud_bool", scaler_type="minmax"):
    """
    Creates a SparkML Pipeline for feature engineering including:
    - StringIndexing and OneHotEncoding for categorical features.
    - VectorAssembly and Scaling (MinMax or Standard) for numerical features.
    - Assembling all processed features into a single vector.

    :param df: Input Spark DataFrame.
    :param input_cols_to_exclude: List of column names to exclude from feature processing (e.g., IDs, label).
    :param label_col: Name of the target/label column (will be excluded from features).
    :param scaler_type: Type of scaler to use ('minmax' or 'standard'). Defaults to 'minmax'.
    :return: A configured SparkML Pipeline object and the final assembled features column name.
    """
    if input_cols_to_exclude is None:
        input_cols_to_exclude = []

    # Add label column to exclusion list if it exists in the DataFrame
    if label_col and label_col in df.columns:
        input_cols_to_exclude.append(label_col)
    input_cols_to_exclude = list(set(input_cols_to_exclude)) # Remove duplicates

    print(f"Columns excluded from feature processing: {input_cols_to_exclude}")

    # Identify categorical and numerical columns automatically based on dtype
    categorical_cols = []
    numerical_cols = []
    numeric_types = (DoubleType, IntegerType, FloatType, LongType, ShortType, ByteType) # Define numeric types

    print("\nIdentifying column types:")
    for field in df.schema.fields:
        col_name = field.name
        if col_name in input_cols_to_exclude:
            print(f"- Skipping excluded column: {col_name}")
            continue # Skip excluded columns

        if isinstance(field.dataType, StringType):
            print(f"- Categorical column found: {col_name}")
            categorical_cols.append(col_name)
        elif isinstance(field.dataType, numeric_types):
            print(f"- Numerical column found: {col_name}")
            numerical_cols.append(col_name)
        else:
             print(f"- Skipping column with other type ({field.dataType}): {col_name}")


    # --- Pipeline Stages ---
    stages = []

    # 1. Categorica   l Feature Processing (StringIndexer + OneHotEncoder)
    for cat_col in categorical_cols:
        # Index string categories to numerical indices
        string_indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index", handleInvalid="keep")
        # One-Hot Encode the indexed categories
        encoder = OneHotEncoder(inputCol=string_indexer.getOutputCol(), outputCol=cat_col + "_vec")
        stages += [string_indexer, encoder]

    # 2. Numerical Feature Processing (VectorAssembler + Scaler)
    if numerical_cols: # Only proceed if there are numerical columns
        # Assemble numerical features into a single vector
        numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features_vec", handleInvalid="keep")
        stages.append(numerical_assembler)

        # Scale the assembled numerical vector
        scaler_output_col = "scaled_numerical_features"
        if scaler_type.lower() == "minmax":
             scaler = MinMaxScaler(inputCol=numerical_assembler.getOutputCol(), outputCol=scaler_output_col)
             print(f"\nUsing MinMaxScaler for numerical features: {numerical_cols}")
        elif scaler_type.lower() == "standard":
             scaler = StandardScaler(inputCol=numerical_assembler.getOutputCol(), outputCol=scaler_output_col, withStd=True, withMean=True)
             print(f"\nUsing StandardScaler for numerical features: {numerical_cols}")
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        stages.append(scaler)
    else:
        scaler_output_col = None # No numerical features to scale
        print("\nNo numerical columns found for scaling.")


    # 3. Assemble Final Feature Vector
    # Combine OHE categorical vectors and scaled numerical vector
    final_feature_cols = [cat_col + "_vec" for cat_col in categorical_cols]
    if scaler_output_col: # Add scaled numerical features if they exist
        final_feature_cols.append(scaler_output_col)

    if not final_feature_cols:
         raise ValueError("No features selected for the final assembly. Check column types and exclusions.")

    print(f"\nColumns included in the final feature vector: {final_feature_cols}")
    final_assembler_output_col = "features"
    final_assembler = VectorAssembler(inputCols=final_feature_cols, outputCol=final_assembler_output_col)
    stages.append(final_assembler)

    # Create the pipeline
    pipeline = Pipeline(stages=stages)
    print(f"\nPipeline created. Final features will be in column: '{final_assembler_output_col}'")

    return pipeline, final_assembler_output_col

if __name__ == "__main__":
    # --- Configuration ---
    HDFS_PARQUET_INPUT_PATH = "hdfs:///user/hs5413_nyu_edu/bank_fraud_data/sample_10pct/" # Input Parquet directory, use 10% dataset to speed up the training
    HDFS_PROCESSED_OUTPUT_PATH = "hdfs:///user/hs5413_nyu_edu/bank_fraud_data/processed_features/" # Output directory for processed data

    # Columns to exclude from feature processing (e.g., unique IDs)
    # Add any columns that are not features and not the label
    COLUMNS_TO_EXCLUDE = ["transaction_id", "account_id", "timestamp"] # Example: Add your specific ID columns
    LABEL_COLUMN = "fraud_bool" # The target variable for your model
    SCALER_TYPE_TO_USE = "minmax" # or "standard"

    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("FeatureEngineeringPipeline") \
        .getOrCreate()

    try:
        # Load data from Parquet
        print(f"Loading Parquet data from: {HDFS_PARQUET_INPUT_PATH}")
        input_df = spark.read.parquet(HDFS_PARQUET_INPUT_PATH)
        print("Input DataFrame Schema:")
        input_df.printSchema()
        input_df.show(5, truncate=False)

        # Create the feature engineering pipeline
        feature_pipeline, final_features_col = create_feature_engineering_pipeline(
            input_df,
            input_cols_to_exclude=COLUMNS_TO_EXCLUDE,
            label_col=LABEL_COLUMN,
            scaler_type=SCALER_TYPE_TO_USE
        )

        # Fit the pipeline to the data (calculates necessary statistics like min/max, indices)
        print("\nFitting the feature engineering pipeline...")
        pipeline_model = feature_pipeline.fit(input_df)

        # Transform the data using the fitted pipeline
        print("Transforming the data...")
        processed_df = pipeline_model.transform(input_df)

        print("\nSchema of processed DataFrame:")
        processed_df.printSchema()

        # Select only relevant columns for saving (optional, saves space)
        # Keep the original label and the final features vector
        columns_to_keep = [LABEL_COLUMN, final_features_col] if LABEL_COLUMN in processed_df.columns else [final_features_col]
        # You might want to keep ID columns as well for joining later
        # columns_to_keep += [col for col in COLUMNS_TO_EXCLUDE if col in processed_df.columns]

        output_df = processed_df.select(*columns_to_keep)

        print(f"\nShowing sample of processed data (columns: {columns_to_keep}):")
        output_df.show(5, truncate=False)

        # Save the processed data (containing the label and the feature vector)
        print(f"\nSaving processed data to: {HDFS_PROCESSED_OUTPUT_PATH}")
        output_df.write.mode("overwrite").parquet(HDFS_PROCESSED_OUTPUT_PATH)
        print("Processed data saved successfully.")

        pipeline_model_path = "hdfs:///user/hs5413_nyu_edu/bank_fraud_data/feature_pipeline_model"
        print(f"Saving pipeline model to: {pipeline_model_path}")
        pipeline_model.save(pipeline_model_path)
        print("Pipeline model saved.")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop the SparkSession
        spark.stop()
        print("\nSparkSession stopped.")
