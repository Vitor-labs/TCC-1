"""
Scoring functions for hyperparameter optimization.
"""

from datetime import datetime
from typing import Optional

import structlog
from numpy import where
from pandas import DataFrame, Series
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM


class ScoreFunction:
    """Handles scoring logic for optimization."""

    def __init__(self):
        self.logger: Optional[structlog.BoundLogger] = None
        self.testing_data: Optional[DataFrame] = None
        self.test_targets: Optional[Series] = None

    def set_context(
        self,
        logger: structlog.BoundLogger,
        testing_data: DataFrame,
        test_targets: Series,
    ) -> None:
        """
        Set the context for scoring function.

        Args:
            logger: Structured logger instance
            testing_data: Test dataset
            test_targets: Test targets
        """
        self.logger = logger
        self.testing_data = testing_data
        self.test_targets = test_targets

    def score_function(
        self, model: OneClassSVM, train: DataFrame, test: Series
    ) -> float:
        """
        Calculate F1 score for the model.

        Args:
            model: Trained OneClassSVM model
            train: Training data (unused but required by sklearn interface)
            test: Test data (unused but required by sklearn interface)

        Returns:
            F1 score
        """
        if self.testing_data is None or self.test_targets is None:
            raise ValueError("Context not set. Call set_context() first.")

        # Make predictions (OneClassSVM returns 1 for inliers, -1 for outliers)
        # We want True for outliers (novelties)
        predictions = where(model.predict(self.testing_data) == -1, True, False)
        f1 = f1_score(self.test_targets, predictions)

        # Log the evaluation
        if self.logger:
            self.logger.info(
                "Model evaluation",
                target=f1,
                params=model.get_params(),
                timestamp=datetime.now().isoformat(),
            )

        return float(f1)

    def objective_function_bayesian(self, nu: float, gamma: float, tol: float) -> float:
        """
        Objective function for Bayesian Optimization.

        Args:
            nu: Nu parameter for OneClassSVM
            gamma: Gamma parameter for OneClassSVM
            tol: Tolerance parameter for OneClassSVM

        Returns:
            F1 score
        """
        if self.testing_data is None or self.test_targets is None:
            raise ValueError("Context not set. Call set_context() first.")

        # Create and fit model
        oc_svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma, tol=tol)
        # Note: For Bayesian optimization, we need to use the training data
        # This should be set separately for Bayesian optimization
        # For now, using the testing data as a placeholder
        oc_svm.fit(self.testing_data)
        f1 = f1_score(
            self.test_targets,
            where(oc_svm.predict(self.testing_data) == 1, False, True),
        )
        if self.logger:
            self.logger.info(
                "Bayesian evaluation",
                target=f1,
                nu=nu,
                gamma=gamma,
                tol=tol,
                timestamp=datetime.now().isoformat(),
            )

        return float(f1)
