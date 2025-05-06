"""
Logging utility to create structured log directories and configure loggers.
Automatically creates timestamped folders for logs, plots, models, and outputs.
"""

import os
import logging

from configs.constants import LOG_DIR

try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Failed to create log directories: {e}")


def get_logger(
    name: str, log_file: str = "log.text", level: int = logging.INFO
) -> logging.Logger:
    """
    Create and return a logger that writes to console, module-specific log file, and a global full.log.

    Args:
        name (str): Name of the logger.
        log_file (str): Filename for the log output file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file), mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        full_log_handler = logging.FileHandler(
            os.path.join(LOG_DIR, "full.log"), mode="a"
        )
        full_log_handler.setFormatter(formatter)
        logger.addHandler(full_log_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
