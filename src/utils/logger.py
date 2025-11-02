"""
Logging configuration and management for hyperparameter optimization.
"""

import logging
from pathlib import Path

import structlog
from config import REPORTS_PATH
from structlog import BoundLogger


class LoggerManager:
    """Manages logging configuration for experiments."""

    @staticmethod
    def configure_structlog() -> None:
        """Configure structured logging for the application."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    @staticmethod
    def configure_file_logger(filepath: str) -> BoundLogger:
        """
        Configure a file-specific logger using structlog.

        Args:
            filepath: Path to the log file

        Returns:
            Configured structlog BoundLogger
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        file_handler = logging.FileHandler(filepath, mode="a")
        file_handler.setLevel(logging.INFO)

        # Configure stdlib logger
        logger_name = f"hyperparameter_search_{Path(filepath).stem}"
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers.clear()
        stdlib_logger.addHandler(file_handler)
        stdlib_logger.setLevel(logging.INFO)

        return structlog.get_logger(logger_name)

    @staticmethod
    def get_log_path(method_name: str) -> str:
        """
        Get the log file path for a specific method.

        Args:
            method_name: Name of the search method

        Returns:
            Log file path
        """
        return str(Path(REPORTS_PATH) / f"logs_{method_name.lower()}.log")


# Initialize structured logging
LoggerManager.configure_structlog()
