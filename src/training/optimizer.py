"""
Main hyperparameter optimization class that coordinates all search methods.
"""

from time import time
from typing import Callable

from bayesian_optimization import BayesianOptimizationSearch
from config import (
    CV_FOLDS,
    NUM_TRIALS,
    RANDOM_STATE,
    VERBOSE_LEVEL,
    ParamGrid,
    SearchMethodType,
)
from data_manager import DataManager
from logger_manager import LoggerManager
from numpy import ndarray
from pandas import DataFrame, Series
from parameter_grid_manager import ParameterGridManager
from score_function import ScoreFunction
from search_algorithms import GeneticAlgorithmSearch, SimulatedAnnealingSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM

from utils import SearchResult


class HyperparameterOptimizer:
    """Main class for hyperparameter optimization experiments."""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.score_function = ScoreFunction()
        self.param_manager = ParameterGridManager()

    def train_search_method(
        self,
        training_data: DataFrame,
        train_targets: Series,
        search_type: SearchMethodType,
        params: ParamGrid,
        scoring: Callable,
        n_iter: int | None = None,
        cv: int = CV_FOLDS,
        verbose: int = VERBOSE_LEVEL,
        random_state: int = RANDOM_STATE,
    ) -> (
        GridSearchCV
        | RandomizedSearchCV
        | SimulatedAnnealingSearch
        | GeneticAlgorithmSearch
        | BayesianOptimizationSearch
    ):
        """
        Train using specified search method.

        Args:
            training_data: Training features
            train_targets: Training targets
            search_type: Type of search method to use
            params: Parameter grid/space
            scoring: Scoring function
            n_iter: Number of iterations (for applicable methods)
            cv: Number of cross-validation folds
            verbose: Verbosity level
            random_state: Random state for reproducibility

        Returns:
            Fitted search object
        """
        if search_type == "SimulatedAnnealing":
            return SimulatedAnnealingSearch(
                param_space=params,
                n_iter=n_iter or NUM_TRIALS,
                random_state=random_state,
            ).fit(training_data, train_targets, scoring)

        elif search_type == "GeneticAlgorithm":
            population_size = min(20, (n_iter or NUM_TRIALS) // 5)
            n_generations = ((n_iter or NUM_TRIALS) // population_size) or 5

            return GeneticAlgorithmSearch(
                param_space=params,
                population_size=population_size,
                n_generations=n_generations,
                random_state=random_state,
            ).fit(training_data, train_targets, scoring)

        elif search_type == "Bayesian":
            return BayesianOptimizationSearch(
                param_space=params,
                random_state=random_state,
            ).fit(training_data, train_targets, n_iter or NUM_TRIALS)

        elif search_type == "Random":
            return RandomizedSearchCV(
                estimator=OneClassSVM(kernel="rbf"),
                param_distributions=params,
                n_iter=n_iter or NUM_TRIALS,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                random_state=random_state,
                error_score="raise",
            ).fit(training_data, train_targets)

        elif search_type == "Grid":
            return GridSearchCV(
                estimator=OneClassSVM(kernel="rbf"),
                param_grid=params,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                error_score="raise",
            ).fit(training_data, train_targets)

        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    def eval_search_method(
        self,
        activities: ndarray,
        search_name: SearchMethodType,
        use_log_dist: bool = False,
    ) -> SearchResult:
        """
        Evaluate a search method across activities using incremental learning.

        Args:
            activities: Array of unique activities
            search_name: Name of search method to evaluate
            use_log_dist: Whether to use log distribution for parameters

        Returns:
            SearchResult containing evaluation metrics
        """
        dist = self.param_manager.get_param_grid(search_name, use_log_dist)
        maximized = False
        best_scores = None

        # Setup logging
        log_path = LoggerManager.get_log_path(search_name)
        logger = LoggerManager.configure_file_logger(log_path)

        start_time = time()

        for i in range(1, len(activities)):
            print(
                f"\nProcessing activity sequence: {activities[:i]} -> {activities[i]}"
            )

            # Update training variables for current activity sequence
            training_data, train_targets, test_data, testing_targets = (
                self.data_manager.update_train_vars(i, activities)
            )

            # Set context for score function
            self.score_function.set_context(logger, test_data, testing_targets)

            print(f"Training for activities {activities[:i]}")

            if not maximized:
                # First iteration: full search
                search_method = self.train_search_method(
                    training_data=training_data,
                    train_targets=train_targets,
                    search_type=search_name,
                    params=dist,
                    scoring=self.score_function.score_function,
                )
                best_scores = search_method
                maximized = True
            else:
                # Subsequent iterations: focused search
                print(f"Already maximized, suggesting new {NUM_TRIALS} points")

                # Update parameter grid for Grid search
                if search_name == "Grid" and hasattr(search_method, "cv_results_"):
                    updated_params = self.param_manager.update_params_grid(
                        search_method.cv_results_, dist
                    )
                else:
                    updated_params = dist

                # Determine number of iterations
                n_iter = (
                    NUM_TRIALS
                    if search_name
                    in ["Random", "SimulatedAnnealing", "GeneticAlgorithm", "Bayesian"]
                    else None
                )

                search_method = self.train_search_method(
                    training_data=training_data,
                    train_targets=train_targets,
                    search_type=search_name,
                    params=updated_params,
                    scoring=self.score_function.score_function,
                    n_iter=n_iter,
                )

                # Update best scores if improvement found
                if search_method.best_score_ > best_scores.best_score_:
                    best_scores = search_method

            print(f"{search_name} Search Best Params: {search_method.best_params_}")
            print(f"{search_name} Search Best Score: {search_method.best_score_:.4f}")

        # Extract CV scores
        cv_scores = best_scores.cv_results_["mean_test_score"]
        if hasattr(cv_scores, "tolist"):
            cv_scores = cv_scores.tolist()

        return SearchResult(
            method=f"{search_name}_search",
            best_params=best_scores.best_params_,
            best_score=best_scores.best_score_,
            cv_scores=cv_scores,
            fit_time=time() - start_time,
            n_evaluations=len(cv_scores),
        )

    def run_experiment(
        self,
        search_methods: list[tuple[SearchMethodType, bool]],
        activities: ndarray | None = None,
    ) -> list[SearchResult]:
        """
        Run hyperparameter optimization experiments for multiple methods.

        Args:
            search_methods: List of (method_name, use_log_dist) tuples
            activities: Activities to use (defaults to all activities)

        Returns:
            List of SearchResult objects
        """
        if activities is None:
            activities = self.data_manager.get_activities()

        results = []

        for method, use_log in search_methods:
            print(f"\n{'=' * 60}")
            print(f"Running {method} search (log_dist={use_log})...")
            print(f"{'=' * 60}")

            try:
                result = self.eval_search_method(activities, method, use_log)
                results.append(result)
                print(
                    f"✓ Completed {method} search. Best score: {result.best_score:.4f}"
                )
            except Exception as e:
                print(f"✗ Error in {method} search: {e}")
                continue

        return results
