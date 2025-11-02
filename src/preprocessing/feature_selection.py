import numpy as np
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import RobustScaler


def apply_variance_filtering(
    X_train: DataFrame, X_test: DataFrame, percentile: float = 10
) -> tuple[DataFrame, DataFrame]:
    """Apply variance-based feature filtering"""
    feature_cols = [
        col
        for col in X_train.columns
        if col not in ["subject", "activityID", "window_start"]
    ]

    if not feature_cols:
        return X_train, X_test

    X_train_features = X_train[feature_cols].copy()
    X_test_features = X_test[feature_cols].copy()

    # Calculate variance threshold (bottom percentile)
    variances = [X_train_features[col].var() for col in feature_cols]
    variance_threshold = np.percentile(variances, percentile)

    variance_selector = VarianceThreshold(threshold=variance_threshold)

    # Fit and transform
    X_train_selected = variance_selector.fit_transform(X_train_features)
    X_test_selected = variance_selector.transform(X_test_features)

    # Get selected feature names
    selected_features = X_train_features.columns[variance_selector.get_support()]

    # Create new DataFrames
    X_train_var = DataFrame(
        X_train_selected, columns=selected_features, index=X_train.index
    )
    X_test_var = DataFrame(
        X_test_selected, columns=selected_features, index=X_test.index
    )

    # Add back metadata columns
    for col in ["subject", "activityID", "window_start"]:
        if col in X_train.columns:
            X_train_var[col] = X_train[col].values
            X_test_var[col] = X_test[col].values

    print(
        f"Variance filtering: {len(feature_cols)} -> {len(selected_features)} features"
    )
    return X_train_var, X_test_var


def apply_correlation_filtering(
    X_train: DataFrame, X_test: DataFrame, threshold: float = 0.95
) -> tuple[DataFrame, DataFrame]:
    """Remove highly correlated features"""
    if not (
        feature_cols := [
            col
            for col in X_train.columns
            if col not in ["subject", "activityID", "window_start"]
        ]
    ):
        return X_train, X_test
    # Calculate correlation matrix
    corr_matrix = X_train[feature_cols].corr().abs()

    # Find highly correlated feature pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    # Remove highly correlated features
    features_to_keep = [
        col
        for col in feature_cols
        if col
        not in [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
    ]
    # Keep all columns (features + metadata)
    all_cols_to_keep = features_to_keep + [
        col for col in X_train.columns if col not in feature_cols
    ]
    X_train_corr = X_train[all_cols_to_keep].copy()
    X_test_corr = X_test[all_cols_to_keep].copy()

    print(
        f"Correlation filtering: {len(feature_cols)} -> {len(features_to_keep)} features"
    )
    return X_train_corr, X_test_corr


def apply_rfe_selection(
    X_train: DataFrame, X_test: DataFrame, y_train: Series, n_features: int = 25
) -> tuple[DataFrame, DataFrame]:
    """Apply Recursive Feature Elimination with Random Forest"""
    feature_cols = [
        col
        for col in X_train.columns
        if col not in ["subject", "activityID", "window_start"]
    ]

    if not feature_cols:
        return X_train, X_test

    X_train_features = X_train[feature_cols].copy()
    X_test_features = X_test[feature_cols].copy()

    # Ensure we don't select more features than available
    n_features_to_select = min(n_features, len(feature_cols))

    # Initialize Random Forest and RFE
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)

    # Fit and transform
    X_train_selected = rfe.fit_transform(X_train_features, y_train)
    X_test_selected = rfe.transform(X_test_features)

    # Get selected feature names
    selected_features = X_train_features.columns[rfe.get_support()]

    # Create new DataFrames
    X_train_rfe = DataFrame(
        X_train_selected, columns=selected_features, index=X_train.index
    )
    X_test_rfe = DataFrame(
        X_test_selected, columns=selected_features, index=X_test.index
    )

    # Add back metadata columns
    for col in ["subject", "activityID", "window_start"]:
        if col in X_train.columns:
            X_train_rfe[col] = X_train[col].values
            X_test_rfe[col] = X_test[col].values

    print(f"RFE selection: {len(feature_cols)} -> {len(selected_features)} features")
    print("Selected features:", list(selected_features))

    return X_train_rfe, X_test_rfe


def normalize_features(
    X_train: DataFrame, X_test: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Normalize features using RobustScaler"""
    feature_cols = [
        col
        for col in X_train.columns
        if col not in ["subject", "activityID", "window_start"]
    ]

    if not feature_cols:
        return X_train, X_test

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Apply RobustScaler to feature columns
    scaler = RobustScaler()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    print("Features normalized using RobustScaler")
    return X_train_scaled, X_test_scaled


def apply_feature_selection_pipeline(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Series,
    n_features: int = 25,
    variance_percentile: float = 10,
    correlation_threshold: float = 0.95,
) -> tuple[DataFrame, DataFrame]:
    """Apply complete feature selection pipeline"""
    print("Applying feature selection pipeline...")
    try:
        print("Filtering features by variance")
        X_train, X_test = apply_variance_filtering(X_train, X_test, variance_percentile)
        print("Filtering features by correaltion")
        X_train, X_test = apply_correlation_filtering(
            X_train, X_test, correlation_threshold
        )
        print("Filtering features with RFE with Random Forest")
        X_train, X_test = apply_rfe_selection(X_train, X_test, y_train, n_features)
        print("Normalizing features")
        X_train, X_test = normalize_features(X_train, X_test)
        print("Feature selection pipeline completed!")
        return X_train, X_test

    except Exception as e:
        raise Exception(f"Error in feature selection pipeline: {e}")
