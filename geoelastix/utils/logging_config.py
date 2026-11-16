"""Logging configuration for GeoElastix."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level="INFO", log_file=None, log_dir=None, log_to_console=True):
    """
    Set up logging configuration for GeoElastix.

    Parameters
    ----------
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
    log_file : str or Path, optional
        Path to log file. If None, only console logging is used.
    log_dir : str or Path, optional
        Directory for log file. If provided with log_file=None, creates
        default log file name with timestamp.
    log_to_console : bool, optional
        Whether to log to console. Default: True

    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("geoelastix")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (if enabled)
    if log_to_console:
        # Windows consoles (cp1252) cannot encode the check marks used in log
        # messages; replace unencodable characters instead of raising
        try:
            sys.stdout.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file is not None or log_dir is not None:
        if log_file is None:
            # Generate default log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"geoelastix_{timestamp}.log"
        else:
            log_file = Path(log_file)

        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger():
    """
    Get the GeoElastix logger instance.

    Returns
    -------
    logger : logging.Logger
        Logger instance
    """
    return logging.getLogger("geoelastix")
