"""
Custom search algorithms for hyperparameter optimization.
"""

import os
import random
from copy import deepcopy
from typing import Callable

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from config import (
    BAYESIAN_INIT_POINTS,
    BAYESIAN_N_ITER,
    GA_CROSSOVER_RATE,
    GA_ELITE_SIZE,
    GA_MUTATION_RATE,
    GA_N_GENERATIONS,
    GA_POPULATION_SIZE,
    NUM_TRIALS,
    RANDOM_STATE,
    REPORTS_PATH,
    SA_COOLING_RATE,
    SA_INITIAL_TEMP,
    SA_MIN_TEMP,
)
from pandas import DataFrame, Series
from score_function import ScoreFunction
from sklearn.model_selection import cross_val_score
from sklearn.svm import OneClassSVM


class BayesianOptimizationSearch:
    """Wrapper for Bayesian Optimization to match the interface of other search methods."""

    def __init__(self, param_space: dict, random_state: int = RANDOM_STATE):
        self.param_space = param_space
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -float("inf")
        self.cv_results_ = {"mean_test_score": []}
        self.optimizer = None
        self.score_function = ScoreFunction()
        self.logs_path = os.path.join(REPORTS_PATH, "logs_bayesian.log")

    def _objective_function(self, nu: float, gamma: float, tol: float) -> float:
        """
        Objective function for Bayesian Optimization.

        Args:
            nu: Nu parameter for OneClassSVM
            gamma: Gamma parameter for OneClassSVM
            tol: Tolerance parameter for OneClassSVM

        Returns:
            F1 score to be maximized
        """
        return self.score_function.objective_function_bayesian(nu, gamma, tol)

    def fit(self, X: DataFrame, y: Series, n_iter: int = BAYESIAN_N_ITER):
        """
        Fit the Bayesian optimization search.

        Args:
            X: Training features
            y: Training targets
            n_iter: Number of optimization iterations

        Returns:
            Self for method chaining
        """
        # Set context for score function (using training data for Bayesian optimization)
        self.score_function.set_context(None, X, y)

        # Initialize Bayesian Optimization
        self.optimizer = BayesianOptimization(
            self._objective_function,
            self.param_space,
            random_state=self.random_state,
        )

        # Setup logging
        if not os.path.exists(self.logs_path):
            os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
            with open(self.logs_path, "w") as fp:
                pass

        self.optimizer.subscribe(
            Events.OPTIMIZATION_STEP, JSONLogger(self.logs_path, reset=False)
        )

        # Load existing logs if available
        if os.path.exists(self.logs_path) and os.path.getsize(self.logs_path) > 0:
            try:
                load_logs(self.optimizer, logs=[self.logs_path])
            except Exception as e:
                print(f"Warning: Could not load existing logs: {e}")

        # Perform optimization
        self.optimizer.maximize(
            init_points=BAYESIAN_INIT_POINTS, n_iter=n_iter - BAYESIAN_INIT_POINTS
        )

        # Extract results
        self.best_params_ = self.optimizer.max["params"]
        self.best_score_ = self.optimizer.max["target"]

        # Extract all scores for cv_results_
        self.cv_results_["mean_test_score"] = [
            res["target"] for res in self.optimizer.res
        ]

        return self

    def get_optimization_history(self) -> dict:
        """
        Get the optimization history.

        Returns:
            dictionary with optimization history
        """
        if self.optimizer is None:
            return {}

        return {
            "params": [res["params"] for res in self.optimizer.res],
            "targets": [res["target"] for res in self.optimizer.res],
            "best_params": self.best_params_,
            "best_score": self.best_score_,
        }


class SimulatedAnnealingSearch:
    """Custom Simulated Annealing implementation for hyperparameter optimization."""

    def __init__(
        self,
        param_space: dict,
        n_iter: int = NUM_TRIALS,
        initial_temp: float = SA_INITIAL_TEMP,
        cooling_rate: float = SA_COOLING_RATE,
        min_temp: float = SA_MIN_TEMP,
        random_state: int = RANDOM_STATE,
    ):
        self.param_space = param_space
        self.n_iter = n_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = {"mean_test_score": []}

        # Set random seeds
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _sample_params(self) -> dict:
        """Sample random parameters from the parameter space."""
        return {
            key: values.rvs(random_state=self.random_state)
            if hasattr(values, "rvs")
            else random.choice(values)
            for key, values in self.param_space.items()
        }

    def _neighbor_params(self, current_params: dict) -> dict:
        """Generate neighboring parameters by slightly modifying current ones."""
        neighbor = deepcopy(current_params)
        param_to_modify = random.choice(list(self.param_space.keys()))

        if hasattr(self.param_space[param_to_modify], "rvs"):
            if param_to_modify == "nu":
                current_val = neighbor[param_to_modify]
                neighbor[param_to_modify] = np.clip(
                    current_val + np.random.normal(0, 0.05 * current_val), 0.001, 1.0
                )
            elif param_to_modify == "gamma":
                neighbor[param_to_modify] = 10 ** np.clip(
                    np.log10(neighbor[param_to_modify]) + np.random.normal(0, 0.1),
                    -4,
                    1,
                )
            elif param_to_modify == "tol":
                neighbor[param_to_modify] = 10 ** np.clip(
                    np.log10(neighbor[param_to_modify]) + np.random.normal(0, 0.1),
                    -6,
                    -1,
                )
        else:
            neighbor[param_to_modify] = random.choice(self.param_space[param_to_modify])

        return neighbor

    def _evaluate_params(
        self, params: dict, X: DataFrame, y: Series, scoring: Callable
    ) -> float:
        """Evaluate parameter configuration using cross-validation."""
        try:
            return np.mean(
                cross_val_score(
                    OneClassSVM(**params, kernel="rbf"), X, y, cv=4, scoring=scoring
                )
            )
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf

    def fit(self, X: DataFrame, y: Series, scoring: Callable):
        """Fit the simulated annealing search."""
        current_params = self._sample_params()
        current_score = self._evaluate_params(current_params, X, y, scoring)

        self.best_params_ = deepcopy(current_params)
        self.best_score_ = current_score
        temperature = self.initial_temp

        for iteration in range(self.n_iter):
            neighbor_params = self._neighbor_params(current_params)
            neighbor_score = self._evaluate_params(neighbor_params, X, y, scoring)
            self.cv_results_["mean_test_score"].append(neighbor_score)

            if neighbor_score > current_score:
                current_params = neighbor_params
                current_score = neighbor_score
            else:
                acceptance_prob = (
                    np.exp((neighbor_score - current_score) / temperature)
                    if temperature > 0
                    else 0
                )
                if random.random() < acceptance_prob:
                    current_params = neighbor_params
                    current_score = neighbor_score

            if current_score > self.best_score_:
                self.best_params_ = deepcopy(current_params)
                self.best_score_ = current_score

            temperature = max(temperature * self.cooling_rate, self.min_temp)

        return self


class GeneticAlgorithmSearch:
    """Custom Genetic Algorithm implementation for hyperparameter optimization."""

    def __init__(
        self,
        param_space: dict,
        population_size: int = GA_POPULATION_SIZE,
        n_generations: int = GA_N_GENERATIONS,
        mutation_rate: float = GA_MUTATION_RATE,
        crossover_rate: float = GA_CROSSOVER_RATE,
        elite_size: int = GA_ELITE_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        self.param_space = param_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = {"mean_test_score": []}

        # Set random seeds
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _create_individual(self) -> dict:
        """Create a random individual (parameter set)."""
        return {
            key: values.rvs(random_state=self.random_state)
            if hasattr(values, "rvs")
            else random.choice(values)
            for key, values in self.param_space.items()
        }

    def _crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        """Create two offspring from two parents using uniform crossover."""
        child1, child2 = deepcopy(parent1), deepcopy(parent2)

        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]

        return child1, child2

    def _mutate(self, individual: dict) -> dict:
        """Mutate an individual by randomly changing some parameters."""
        mutated = deepcopy(individual)

        for key in individual.keys():
            if random.random() < self.mutation_rate:
                mutated[key] = (
                    self.param_space[key].rvs(random_state=self.random_state)
                    if hasattr(self.param_space[key], "rvs")
                    else random.choice(self.param_space[key])
                )
        return mutated

    def _tournament_selection(
        self, population: list, fitness_scores: list, tournament_size: int = 3
    ) -> dict:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )
        return population[
            tournament_indices[
                np.argmax([fitness_scores[i] for i in tournament_indices])
            ]
        ]

    def _evaluate_params(
        self, params: dict, X: DataFrame, y: Series, scoring: Callable
    ) -> float:
        """Evaluate parameter configuration using cross-validation."""
        try:
            return np.mean(
                cross_val_score(
                    OneClassSVM(**params, kernel="rbf"), X, y, cv=4, scoring=scoring
                )
            )
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf

    def fit(self, X: DataFrame, y: Series, scoring: Callable):
        """Fit the genetic algorithm search."""
        population = [self._create_individual() for _ in range(self.population_size)]

        for generation in range(self.n_generations):
            print(f"Evaluating Generation {generation}")
            fitness_scores = []

            for individual in population:
                score = self._evaluate_params(individual, X, y, scoring)
                fitness_scores.append(score)
                self.cv_results_["mean_test_score"].append(score)

                if score > self.best_score_:
                    self.best_params_ = deepcopy(individual)
                    self.best_score_ = score

            # Create new population
            new_population = [
                deepcopy(population[idx])
                for idx in np.argsort(fitness_scores)[-self.elite_size :]
            ]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                new_population.extend([self._mutate(child1), self._mutate(child2)])

            population = new_population[: self.population_size]

        return self
