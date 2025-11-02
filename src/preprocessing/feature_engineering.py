import numpy as np
from pandas import DataFrame
from scipy.stats import kurtosis, skew


def extract_statistical_features(window_data: DataFrame, prefix: str = "feat") -> dict:
    """Extract statistical features from a time window"""
    features = {}

    for col in window_data.columns:
        if col.startswith("IMU_"):
            data = window_data[col].values
            features[f"{prefix}_{col}_mean"] = np.mean(data)
            features[f"{prefix}_{col}_std"] = np.std(data)
            features[f"{prefix}_{col}_var"] = np.var(data)
            features[f"{prefix}_{col}_min"] = np.min(data)
            features[f"{prefix}_{col}_max"] = np.max(data)
            features[f"{prefix}_{col}_range"] = np.max(data) - np.min(data)
            features[f"{prefix}_{col}_rms"] = np.sqrt(np.mean(data**2))
            features[f"{prefix}_{col}_skew"] = skew(data)
            features[f"{prefix}_{col}_kurtosis"] = kurtosis(data)
            features[f"{prefix}_{col}_q25"] = np.percentile(data, 25)
            features[f"{prefix}_{col}_q75"] = np.percentile(data, 75)

    return features


def extract_temporal_features(window_data: DataFrame, prefix: str = "feat") -> dict:
    """Extract temporal/sequential features"""
    features = {}

    for col in window_data.columns:
        if col.startswith("IMU_"):
            data = window_data[col].values

            # Zero crossing rate
            if len(data) > 1:
                zero_crossings = np.sum(np.diff(np.signbit(data)))
                features[f"{prefix}_{col}_zcr"] = zero_crossings / len(data)
            else:
                features[f"{prefix}_{col}_zcr"] = 0

            # Signal energy
            features[f"{prefix}_{col}_energy"] = np.sum(data**2)

            # Autocorrelation at lag 1
            if len(data) > 1:
                try:
                    autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                    features[f"{prefix}_{col}_autocorr"] = (
                        autocorr if not np.isnan(autocorr) else 0
                    )
                except:
                    features[f"{prefix}_{col}_autocorr"] = 0
            else:
                features[f"{prefix}_{col}_autocorr"] = 0

    return features


def extract_frequency_features(
    window_data: DataFrame, prefix: str = "feat", sampling_rate: int = 100
) -> dict:
    """Extract frequency domain features"""
    features = {}

    for col in window_data.columns:
        if col.startswith("IMU_"):
            data = window_data[col].values

            if len(data) < 4:  # Need minimum length for FFT
                features[f"{prefix}_{col}_dominant_freq"] = 0
                features[f"{prefix}_{col}_spectral_energy"] = 0
                features[f"{prefix}_{col}_spectral_centroid"] = 0
                continue

            try:
                # FFT-based features
                fft = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data), 1 / sampling_rate)

                # Power spectral density
                psd = np.abs(fft) ** 2

                # Dominant frequency (skip DC component)
                if len(psd) > 2:
                    dominant_freq_idx = np.argmax(psd[1 : len(psd) // 2]) + 1
                    features[f"{prefix}_{col}_dominant_freq"] = abs(
                        freqs[dominant_freq_idx]
                    )
                else:
                    features[f"{prefix}_{col}_dominant_freq"] = 0

                # Spectral energy
                features[f"{prefix}_{col}_spectral_energy"] = np.sum(psd)

                # Spectral centroid
                positive_freqs = freqs[: len(freqs) // 2]
                positive_psd = psd[: len(psd) // 2]
                if np.sum(positive_psd) > 0:
                    features[f"{prefix}_{col}_spectral_centroid"] = np.sum(
                        positive_freqs * positive_psd
                    ) / np.sum(positive_psd)
                else:
                    features[f"{prefix}_{col}_spectral_centroid"] = 0

            except Exception:
                # Fallback values if FFT fails
                features[f"{prefix}_{col}_dominant_freq"] = 0
                features[f"{prefix}_{col}_spectral_energy"] = np.sum(
                    data**2
                )  # Use signal energy as fallback
                features[f"{prefix}_{col}_spectral_centroid"] = 0

    return features


def extract_movement_features(window_data: DataFrame, prefix: str = "feat") -> dict:
    """Extract movement-specific features"""
    features = {}

    # Extract accelerometer and gyroscope data
    sensors = ["hand", "chest", "ankle"]

    for sensor in sensors:
        # Get sensor-specific columns
        sensor_cols = [col for col in window_data.columns if f"IMU_{sensor}" in col]

        if len(sensor_cols) >= 10:  # Ensure we have enough columns
            # Accelerometer magnitude (columns 2-4 for each sensor, 0-indexed)
            acc_cols = sensor_cols[1:4]  # Skip temperature, take first 3D accel
            if len(acc_cols) == 3:
                try:
                    acc_data = window_data[acc_cols].values
                    # Magnitude vector
                    magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
                    features[f"{prefix}_{sensor}_acc_magnitude_mean"] = np.mean(
                        magnitude
                    )
                    features[f"{prefix}_{sensor}_acc_magnitude_std"] = np.std(magnitude)

                    # Jerk (rate of change of acceleration)
                    if len(magnitude) > 1:
                        jerk = np.diff(magnitude)
                        features[f"{prefix}_{sensor}_jerk_mean"] = np.mean(jerk)
                        features[f"{prefix}_{sensor}_jerk_std"] = np.std(jerk)
                    else:
                        features[f"{prefix}_{sensor}_jerk_mean"] = 0
                        features[f"{prefix}_{sensor}_jerk_std"] = 0
                except:
                    # Fallback values
                    features[f"{prefix}_{sensor}_acc_magnitude_mean"] = 0
                    features[f"{prefix}_{sensor}_acc_magnitude_std"] = 0
                    features[f"{prefix}_{sensor}_jerk_mean"] = 0
                    features[f"{prefix}_{sensor}_jerk_std"] = 0

            # Gyroscope magnitude (columns 7-9 for each sensor)
            if len(sensor_cols) >= 10:
                gyro_cols = sensor_cols[7:10]
                if len(gyro_cols) == 3:
                    try:
                        gyro_data = window_data[gyro_cols].values
                        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
                        features[f"{prefix}_{sensor}_gyro_magnitude_mean"] = np.mean(
                            gyro_magnitude
                        )
                        features[f"{prefix}_{sensor}_gyro_magnitude_std"] = np.std(
                            gyro_magnitude
                        )
                    except:
                        features[f"{prefix}_{sensor}_gyro_magnitude_mean"] = 0
                        features[f"{prefix}_{sensor}_gyro_magnitude_std"] = 0

    return features


def extract_cross_sensor_features(window_data: DataFrame, prefix: str = "feat") -> dict:
    """Extract cross-sensor correlation features"""
    features = {}

    sensors = ["hand", "chest", "ankle"]
    sensor_data = {}

    # Get magnitude vectors for each sensor
    for sensor in sensors:
        sensor_cols = [col for col in window_data.columns if f"IMU_{sensor}" in col]
        if len(sensor_cols) >= 4:
            acc_cols = sensor_cols[1:4]  # First accelerometer data
            if len(acc_cols) == 3:
                try:
                    acc_data = window_data[acc_cols].values
                    sensor_data[sensor] = np.sqrt(np.sum(acc_data**2, axis=1))
                except:
                    continue

    # Cross-correlations between sensors
    sensor_pairs = [("hand", "chest"), ("hand", "ankle"), ("chest", "ankle")]
    for sensor1, sensor2 in sensor_pairs:
        if sensor1 in sensor_data and sensor2 in sensor_data:
            try:
                correlation = np.corrcoef(sensor_data[sensor1], sensor_data[sensor2])[
                    0, 1
                ]
                features[f"{prefix}_corr_{sensor1}_{sensor2}"] = (
                    correlation if not np.isnan(correlation) else 0
                )
            except:
                features[f"{prefix}_corr_{sensor1}_{sensor2}"] = 0
        else:
            features[f"{prefix}_corr_{sensor1}_{sensor2}"] = 0

    return features


def extract_window_features(window_data: DataFrame, window_id: int = 0) -> dict:
    """Extract all features from a time window with consistent naming"""
    # Use consistent prefix for all windows
    prefix = "feat"

    features = {}
    features.update(extract_statistical_features(window_data, prefix))
    features.update(extract_temporal_features(window_data, prefix))
    features.update(extract_frequency_features(window_data, prefix))
    features.update(extract_movement_features(window_data, prefix))
    features.update(extract_cross_sensor_features(window_data, prefix))

    return features


def create_windowed_features(
    df: DataFrame, window_size: int = 300, overlap: float = 0.5
) -> DataFrame:
    """Create features from overlapping windows"""
    print("Extracting features from time windows...")
    step_size = int(window_size * (1 - overlap))
    features_list = []
    window_count = 0

    # Group by subject and activity to maintain temporal coherence
    for (subject, activity), group in df.groupby(["subject", "activityID"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        # Create windows only if we have enough data
        if len(group) >= window_size:
            for i in range(0, len(group) - window_size + 1, step_size):
                window_data = group.iloc[i : i + window_size]
                # Extract features with consistent naming
                window_features = extract_window_features(window_data, window_count)
                window_features["subject"] = subject
                window_features["activityID"] = activity
                window_features["window_start"] = i
                window_features["window_id"] = window_count

                features_list.append(window_features)
                window_count += 1

    if features_list:
        features_df = DataFrame(features_list)
        print(
            f"Created {len(features_df)} feature windows with {len(features_df.columns) - 4} features each"
        )
        # Handle any remaining NaN or infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)

        return features_df
    else:
        print(
            "Warning: No feature windows created. Check data size and window parameters."
        )
        return DataFrame()
