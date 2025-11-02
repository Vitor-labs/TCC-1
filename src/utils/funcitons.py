"""
Utility functions for hyperparameter optimization experiments.
"""

from dataclasses import dataclass
from json import dumps
from pathlib import Path
from typing import Dict, List

import numpy as np
from config import CONFIG_PATH
from scipy.stats import wilcoxon


@dataclass
class SearchResult:
    """Container for hyperparameter search results."""

    method: str
    best_params: Dict
    best_score: float
    cv_scores: List[float]
    fit_time: float
    n_evaluations: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_scores": self.cv_scores,
            "fit_time": self.fit_time,
            "n_evaluations": self.n_evaluations,
        }


def compare_search_methods(
    scores1: List[float], scores2: List[float]
) -> Dict[str, float]:
    """
    Statistical comparison of search methods using Wilcoxon signed-rank test.

    Args:
        scores1: First set of scores
        scores2: Second set of scores

    Returns:
        Dictionary with comparison statistics
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have equal length")

    # Handle case where all scores are identical
    if len(set(scores1 + scores2)) == 1:
        return {
            "wilcoxon_statistic": 0.0,
            "p_value": 1.0,
            "scores1_mean": np.mean(scores1),
            "scores2_mean": np.mean(scores2),
            "effect_size": 0.0,
        }

    try:
        statistic, p_value = wilcoxon(scores1, scores2, alternative="two-sided")
    except ValueError:
        # Handle case where differences are all zero
        statistic, p_value = 0.0, 1.0

    # Calculate effect size (Cohen's d approximation)
    pooled_std = np.std(scores1 + scores2)
    effect_size = (
        (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0.0
    )

    return {
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "scores1_mean": float(np.mean(scores1)),
        "scores2_mean": float(np.mean(scores2)),
        "effect_size": float(effect_size),
    }


def generate_all_comparisons(results: List[SearchResult]) -> List[Dict]:
    """
    Generate all pairwise comparisons between search methods.

    Args:
        results: List of SearchResult objects

    Returns:
        List of comparison dictionaries
    """
    comparisons = []

    for i, result1 in enumerate(results):
        for j, result2 in enumerate(results[i + 1 :], i + 1):
            try:
                comparison = {
                    "test": f"{result1.method} vs {result2.method}",
                    **compare_search_methods(result1.cv_scores, result2.cv_scores),
                }
                comparisons.append(comparison)
            except Exception as e:
                print(
                    f"Warning: Could not compare {result1.method} vs {result2.method}: {e}"
                )
                continue

    return comparisons


def save_results(results: List[SearchResult], comparisons: List[Dict]) -> None:
    """
    Save experiment results to files.

    Args:
        results: List of SearchResult objects
        comparisons: List of comparison dictionaries
    """
    # Ensure output directory exists
    Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)

    # Save comparison results
    comparison_path = Path(CONFIG_PATH) / "test_results.json"
    with open(comparison_path, "w") as file:
        file.write(dumps(comparisons, indent=2))

    # Save individual results
    results_path = Path(CONFIG_PATH) / "search_results.json"
    results_dict = {result.method: result.to_dict() for result in results}
    with open(results_path, "w") as file:
        file.write(dumps(results_dict, indent=2))

    print(f"Results saved to {comparison_path} and {results_path}")


def load_results(results_path: str = None) -> Dict:
    """
    Load previously saved results.

    Args:
        results_path: Path to results file

    Returns:
        Dictionary with loaded results
    """
    if results_path is None:
        results_path = Path(CONFIG_PATH) / "search_results.json"

    try:
        with open(results_path, "r") as file:
            return eval(file.read())  # Note: In production, use json.load instead
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        return {}


def print_summary(results: List[SearchResult]) -> None:
    """
    Print a summary of all results.

    Args:
        results: List of SearchResult objects
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Sort by best score
    sorted_results = sorted(results, key=lambda x: x.best_score, reverse=True)

    print(f"{'Method':<25} {'Best Score':<12} {'Fit Time (s)':<12} {'Evaluations':<12}")
    print("-" * 65)

    for result in sorted_results:
        print(
            f"{result.method:<25} {result.best_score:<12.4f} "
            f"{result.fit_time:<12.2f} {result.n_evaluations:<12}"
        )

    print("-" * 65)
    print(f"Best performing method: {sorted_results[0].method}")
    print(f"Best score achieved: {sorted_results[0].best_score:.4f}")
