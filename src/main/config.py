"""
Configuration settings and constants for hyperparameter optimization.
"""

from typing import Final, Literal, Union

from scipy.stats import loguniform

# Type definitions
ParamGrid = dict[str, Union[tuple[float, ...], tuple[str, ...], list]]
SearchMethodType = Literal[
    "Grid", "Random", "SimulatedAnnealing", "GeneticAlgorithm", "Bayesian"
]

# Constants
NUM_TRIALS: Final[int] = 10
RANDOM_STATE: Final[int] = 42
DEFAULT_DATA_PATH: Final[str] = "../data/PAMAP2/"
REPORTS_PATH: Final[str] = "../reports/"
CONFIG_PATH: Final[str] = "../conf/"

# Hyperparameter grids
PARAM_GRIDS = {
    "Grid": {
        "nu": [0.01, 0.05, 0.1, 0.25],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "tol": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    },
    "Random": {
        "nu": [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "Random_log": {
        "nu": loguniform(0.001, 0.3),
        "gamma": loguniform(1e-4, 10),
        "tol": loguniform(1e-6, 1e-1),
    },
    "Bayesian": {"nu": (0.01, 0.5), "gamma": (1e-4, 1), "tol": (1e-5, 1e-1)},
}

# Cross-validation settings
CV_FOLDS: Final[int] = 4
VERBOSE_LEVEL: Final[int] = 1

# Simulated Annealing settings
SA_INITIAL_TEMP: Final[float] = 1.0
SA_COOLING_RATE: Final[float] = 0.95
SA_MIN_TEMP: Final[float] = 0.01

# Genetic Algorithm settings
GA_POPULATION_SIZE: Final[int] = 20
GA_N_GENERATIONS: Final[int] = 10
GA_MUTATION_RATE: Final[float] = 0.1
GA_CROSSOVER_RATE: Final[float] = 0.8
GA_ELITE_SIZE: Final[int] = 2

# Bayesian Optimization settings
BAYESIAN_INIT_POINTS: Final[int] = 5
BAYESIAN_N_ITER: Final[int] = 75
