import os

from feature_engineering import create_windowed_features
from feature_selection import apply_feature_selection_pipeline
from pandas import DataFrame, Series, concat, read_csv, set_option, to_datetime

set_option("display.max_columns", None)


def read_w_log(path: str, filename: str) -> tuple[DataFrame, str]:
    """Read data file with logging"""
    print(f"Reading: {filename}")
    return (
        read_csv(os.path.join(path, filename), sep=r"\s+", header=None),
        filename.split(".")[0][-2:],
    )


def handle_missing_data(df: DataFrame) -> DataFrame:
    """Handle NaN values with appropriate strategies"""
    # For IMU data: linear interpolation for short gaps, drop for long gaps
    for col in [col for col in df.columns if col.startswith("IMU_")]:
        # Only interpolate if gap is ≤ 5 samples (0.05s at 100Hz)
        df.loc[:, col] = df[col].interpolate("linear", limit=5, limit_direction="both")

    return df.dropna()


def optimize_data_types(df: DataFrame) -> DataFrame:
    """Optimize data types for memory efficiency"""
    # Convert categorical columns
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype("category")
    if "activityID" in df.columns:
        df["activityID"] = df["activityID"].astype("int8")
    if "timestamp" in df.columns:
        df["timestamp"] = to_datetime(df["timestamp"], unit="s")

    # Convert float64 to float32 for IMU data
    float_columns = df.select_dtypes(include=["float64"]).columns
    if len(float_columns) > 0:
        df[float_columns] = df[float_columns].astype("float32")

    return df


def load_raw_data(path: str, test_size: float = 0.2) -> tuple[DataFrame, DataFrame]:
    """Load and preprocess raw PAMAP2 data"""
    print("Loading raw PAMAP2 data...")

    train_data, test_data = [], []
    column_names = (
        ["timestamp", "activityID", "heart_rate"]
        + [f"IMU_hand_{i}" for i in range(1, 18)]
        + [f"IMU_chest_{i}" for i in range(1, 18)]
        + [f"IMU_ankle_{i}" for i in range(1, 18)]
    )
    for df, subject in [  # Process each data file
        read_w_log(path, filename)
        for filename in os.listdir(path)
        if filename.endswith(".dat")
    ]:
        df.columns = column_names
        # Handle missing data and filter out problematic activities
        df = handle_missing_data(df[df["activityID"] != 0])
        df["subject"] = subject
        # Remove problematic columns (orientation data columns 15-17)
        df = df.loc[:, ~df.columns.str.endswith(("_15", "_16", "_17"))]
        # Optimize data types
        df = optimize_data_types(df)

        # Split by time for each activity to maintain temporal order
        for label in df["activityID"].unique():
            activity_data = df[df["activityID"] == label].sort_values("timestamp")

            if len(activity_data) > 0:  # Only process if we have data
                split_idx = int((1 - test_size) * len(activity_data))

                train_data.append(activity_data[:split_idx])
                test_data.append(activity_data[split_idx:])

    # Combine all data
    train_df = concat(train_data, ignore_index=True) if train_data else DataFrame()
    test_df = concat(test_data, ignore_index=True) if test_data else DataFrame()

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    return train_df, test_df


def process_data(
    data_path: str,
    output_path: str,
    test_size: float = 0.2,
    window_size: int = 300,
    window_overlap: float = 0.5,
    n_features: int = 25,
) -> tuple[DataFrame, DataFrame, Series, Series]:
    """Complete data processing pipeline"""

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Step 1: Load raw data
    train_df, test_df = load_raw_data(data_path, test_size)

    if train_df.empty or test_df.empty:
        raise ValueError("No data loaded. Check data path and file format.")

    # Step 2: Extract features from time windows
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    train_features = create_windowed_features(train_df, window_size, window_overlap)
    test_features = create_windowed_features(test_df, window_size, window_overlap)

    if train_features.empty or test_features.empty:
        raise ValueError("No features extracted. Check window size and data length.")

    # Step 3: Separate features and targets
    y_train = train_features["activityID"]
    y_test = test_features["activityID"]
    # Step 4: Apply feature selection pipeline
    print("\n" + "=" * 50)
    print("FEATURE SELECTION")
    print("=" * 50)

    X_train, X_test = apply_feature_selection_pipeline(
        train_features, test_features, y_train, n_features=n_features
    )
    # Step 5: Save processed data
    print("\n" + "=" * 50)
    print("SAVING PROCESSED DATA")
    print("=" * 50)

    X_train.to_csv(os.path.join(output_path, "x_train_features.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "x_test_features.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train_features.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test_features.csv"), index=False)

    print(f"Data saved to {output_path}")
    print(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def split_data(test_size: float = 0.2) -> tuple[DataFrame, DataFrame, Series, Series]:
    """Main function for data splitting with feature engineering and selection"""

    data_path = "./data/PAMAP2_Dataset/Protocol/"
    output_path = "./data/PAMAP2"

    return process_data(
        data_path=data_path,
        output_path=output_path,
        test_size=test_size,
        window_size=300,  # 3 seconds at 100Hz
        window_overlap=0.5,  # 50% overlap
        n_features=25,  # Final number of features
    )


if __name__ == "__main__":
    print("PAMAP2 Data Processing Pipeline")
    print("=" * 50)

    try:
        X_train, X_test, y_train, y_test = split_data()

        print("\n" + "=" * 50)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Train Shape: {X_train.shape}")
        print(f"Test Shape: {X_test.shape}")
        print(f"Unique activities in train: {sorted(y_train.unique())}")
        print(f"Unique activities in test: {sorted(y_test.unique())}")

        # Show final feature columns
        feature_cols = [
            col
            for col in X_train.columns
            if col not in ["subject", "activityID", "window_start"]
        ]
        print(f"\nFinal feature set ({len(feature_cols)} features):")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
