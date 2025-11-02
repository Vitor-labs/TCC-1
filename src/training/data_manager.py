"""
Data management utilities for hyperparameter optimization experiments.
"""

from pathlib import Path
from typing import Tuple

from config import DEFAULT_DATA_PATH, RANDOM_STATE
from numpy import ndarray
from pandas import DataFrame, Series, concat, read_csv


class DataManager:
    """Handles data loading and preprocessing operations."""

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = Path(data_path)
        self._load_data()
        self._prepare_data()

    def _load_data(self) -> None:
        """Load training and testing datasets."""
        try:
            self.X_train = read_csv(self.data_path / "x_train_data.csv")
            self.X_test = read_csv(self.data_path / "x_test_data.csv")
            self.y_train = read_csv(self.data_path / "y_train_data.csv")
            self.y_test = read_csv(self.data_path / "y_test_data.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data files not found in {self.data_path}: {e}")

    def _prepare_data(self) -> None:
        """Prepare data by adding activity labels."""
        self.X_train["activity"] = self.y_train.iloc[:, 0]
        self.X_test["activity"] = self.y_test.iloc[:, 0]
        self.min_samples = self.X_train["activity"].value_counts().min()

    def get_activities(self) -> ndarray:
        """Get unique activities from training data."""
        return self.X_train["activity"].unique()

    def update_train_vars(
        self, i: int, activities: ndarray
    ) -> Tuple[DataFrame, Series, DataFrame, Series]:
        """
        Update training variables for activity sequence.

        Args:
            i: Current activity index
            activities: Array of unique activities

        Returns:
            Tuple of (training_data, train_targets, testing_data, test_targets)
        """
        training = (
            self.X_train[self.X_train["activity"].isin(activities[:i])]
            .groupby("activity")
            .head(self.min_samples)
        )
        testing = self.X_test[self.X_test["activity"] == activities[i]].head(
            self.min_samples
        )
        training = training.copy()
        testing = testing.copy()
        training.loc[:, "isNovelty"] = False
        testing.loc[:, "isNovelty"] = True

        novelty = concat(
            [
                testing,
                training.sample(n=int(0.15 * len(training)), random_state=RANDOM_STATE),
            ]
        )
        return (
            training.drop(columns=["activity", "isNovelty"]),
            training["isNovelty"],
            novelty.drop(columns=["activity", "isNovelty"]),
            novelty["isNovelty"],
        )

    def get_data_info(self) -> dict:
        """Get information about the loaded data."""
        return {
            "train_shape": self.X_train.shape,
            "test_shape": self.X_test.shape,
            "activities": list(self.get_activities()),
            "min_samples_per_activity": self.min_samples,
        }
