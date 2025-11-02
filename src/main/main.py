"""
Main script for running hyperparameter optimization experiments.
"""

import argparse
from pathlib import Path

from config import DEFAULT_DATA_PATH
from data_manager import DataManager
from hyperparameter_optimizer import HyperparameterOptimizer

from utils import generate_all_comparisons, print_summary, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization Experiments"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to data directory",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Grid", "Random", "SimulatedAnnealing"],
        help="Search methods to run",
    )
    parser.add_argument(
        "--use-log-dist",
        action="store_true",
        help="Use log distributions for continuous parameters",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with fewer iterations",
    )
    return parser.parse_args()


def validate_setup(data_path: str) -> bool:
    """
    Validate that the experimental setup is correct.

    Args:
        data_path: Path to data directory

    Returns:
        True if setup is valid, False otherwise
    """
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return False

    for file in [
        "x_train_data.csv",
        "x_test_data.csv",
        "y_train_data.csv",
        "y_test_data.csv",
    ]:
        if not (data_dir / file).exists():
            print(f"Error: Required file not found: {file}")
            return False

    return True


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION EXPERIMENTS")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Methods: {args.methods}")
    print(f"Use log distributions: {args.use_log_dist}")
    print(f"Quick test mode: {args.quick_test}")
    print("=" * 80)

    # Validate setup
    if not validate_setup(args.data_path):
        print("Setup validation failed. Exiting...")
        return 1

    try:
        # Initialize components
        print("\nInitializing data manager...")
        data_manager = DataManager(args.data_path)

        print("Data loaded successfully:")
        data_info = data_manager.get_data_info()
        for key, value in data_info.items():
            print(f"  {key}: {value}")

        print("\nInitializing hyperparameter optimizer...")
        optimizer = HyperparameterOptimizer(data_manager)
        activities = data_manager.get_activities()

        # Prepare search methods
        search_methods = []
        for method in args.methods:
            if method in ["Random", "SimulatedAnnealing", "GeneticAlgorithm"]:
                # Add both regular and log-distributed versions for these methods
                search_methods.append((method, False))
                if args.use_log_dist:
                    search_methods.append((method, True))
            else:
                search_methods.append((method, False))

        print(f"\nWill run {len(search_methods)} experiments:")
        for method, use_log in search_methods:
            print(f"  - {method} (log_dist={use_log})")

        # Run experiments
        print("\nStarting experiments...")
        results = optimizer.run_experiment(search_methods, activities)

        if not results:
            print("No results obtained. Check for errors above.")
            return 1

        # Generate comparisons
        print("\nGenerating statistical comparisons...")
        comparisons = generate_all_comparisons(results)

        # Save results
        print("\nSaving results...")
        save_results(results, comparisons)

        # Print summary
        print_summary(results)

        # Print some comparison highlights
        if comparisons:
            print("\n" + "=" * 80)
            print("STATISTICAL COMPARISON HIGHLIGHTS")
            print("=" * 80)

            significant_comparisons = [
                comp for comp in comparisons if comp.get("p_value", 1.0) < 0.05
            ]

            if significant_comparisons:
                print(
                    f"Found {len(significant_comparisons)} statistically significant differences:"
                )
                for comp in significant_comparisons[:5]:  # Show top 5
                    print(
                        f"  {comp['test']}: p={comp['p_value']:.4f}, "
                        f"effect_size={comp['effect_size']:.4f}"
                    )
            else:
                print("No statistically significant differences found (p < 0.05)")

        print("\n✓ Experiments completed successfully!")
        print(f"✓ Results saved to {Path('../conf/').absolute()}")

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
