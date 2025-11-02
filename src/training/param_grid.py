"""
Parameter grid management for hyperparameter optimization.
"""

from copy import deepcopy
from typing import Dict

from config import NUM_TRIALS, PARAM_GRIDS, ParamGrid, SearchMethodType
from pandas import DataFrame


class ParameterGridManager:
    """Manages parameter grids for different search methods."""

    @staticmethod
    def get_param_grid(
        search_method: SearchMethodType, use_log_dist: bool = False
    ) -> ParamGrid:
        """
        Get parameter grid based on search method.

        Args:
            search_method: Type of search method
            use_log_dist: Whether to use log distribution for continuous parameters

        Returns:
            Parameter grid dictionary
        """
        if search_method == "Grid":
            return deepcopy(PARAM_GRIDS["Grid"])

        elif search_method in ["Random", "SimulatedAnnealing", "GeneticAlgorithm"]:
            if use_log_dist:
                return deepcopy(PARAM_GRIDS["Random_log"])
            else:
                return deepcopy(PARAM_GRIDS["Random"])

        elif search_method == "Bayesian":
            return deepcopy(PARAM_GRIDS["Bayesian"])

        else:
            raise ValueError(f"Unknown search method: {search_method}")

    @staticmethod
    def update_params_grid(cv_results: Dict, og_param_grid: ParamGrid) -> ParamGrid:
        """
        Update parameter grid based on cross-validation results.

        Args:
            cv_results: Cross-validation results from search
            og_param_grid: Original parameter grid

        Returns:
            Updated parameter grid
        """
        params = ["gamma", "nu", "tol"]

        # Create DataFrame from CV results
        top_entries = (
            DataFrame(
                {
                    "rank_test_score": cv_results["rank_test_score"],
                    "gamma": cv_results["param_gamma"],
                    "nu": cv_results["param_nu"],
                    "tol": cv_results["param_tol"],
                }
            )
            .sort_values("rank_test_score")
            .head(NUM_TRIALS)
        )

        # Diversify if stagnation detected
        if len(top_entries) == NUM_TRIALS:
            print("Detected potential parameter space stagnation, diversifying...")

            current_params = {col: set(top_entries[col]) for col in params}
            unused_params = {
                param: list(set(og_param_grid[param]) - current_params[param])
                for param in params
            }

            # Add unused parameters to diversify
            for param in params:
                if unused_params[param]:
                    current_unique = list(dict.fromkeys(top_entries[param]))
                    if len(current_unique) > 1 and unused_params[param]:
                        current_unique = current_unique[:-1] + [unused_params[param][0]]
                    elif unused_params[param]:
                        current_unique.append(unused_params[param][0])

                    # Update the last occurrence
                    mask = (
                        top_entries[param]
                        == list(dict.fromkeys(top_entries[param]))[-1]
                    )
                    top_entries.loc[mask, param] = unused_params[param][0]

        # Create result dictionary
        result = {col: list(dict.fromkeys(top_entries[col])) for col in params}
        cartesian_size = len(result["gamma"]) * len(result["nu"]) * len(result["tol"])

        # Reduce parameter space if too large
        while cartesian_size > NUM_TRIALS:
            best_reduction = None
            best_param = None

            for param in params:
                if len(result[param]) > 1:
                    temp_sizes = [
                        len(result[p]) if p != param else len(result[p]) - 1
                        for p in params
                    ]
                    new_size = temp_sizes[0] * temp_sizes[1] * temp_sizes[2]
                    if new_size >= NUM_TRIALS and (
                        best_reduction is None or new_size < best_reduction
                    ):
                        best_reduction = new_size
                        best_param = param

            if best_param is None:
                param_lengths = [(param, len(result[param])) for param in params]
                param_lengths.sort(key=lambda x: x[1], reverse=True)
                best_param = param_lengths[0][0]

            if len(result[best_param]) > 1:
                result[best_param] = result[best_param][:-1]

            cartesian_size = (
                len(result["gamma"]) * len(result["nu"]) * len(result["tol"])
            )

            if all(len(result[param]) == 1 for param in params):
                break

        print(f"Parameter grid of size {cartesian_size}: {result}")
        return result

    @staticmethod
    def validate_param_grid(
        param_grid: ParamGrid, search_method: SearchMethodType
    ) -> bool:
        """
        Validate parameter grid for a specific search method.

        Args:
            param_grid: Parameter grid to validate
            search_method: Search method type

        Returns:
            True if valid, False otherwise
        """
        required_params = {"nu", "gamma", "tol"}

        if not required_params.issubset(param_grid.keys()):
            return False

        # Additional validation based on search method
        if search_method == "Bayesian":
            # Check if all values are tuples with 2 elements
            return all(
                isinstance(value, tuple) and len(value) == 2
                for value in param_grid.values()
            )

        return True
