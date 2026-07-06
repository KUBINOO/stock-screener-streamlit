"""
logger_config.py
Centralized logging configuration for the Streamlit application.
Routes all logs strictly to sys.stdout for cloud compatibility and structured log aggregation.
"""

import logging
import sys

_FORMATTER_STRING = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DEFAULT_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves or creates a configured logger instance with sys.stdout StreamHandler.
    Ensures duplicate handlers are avoided when called repeatedly across modules.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(_DEFAULT_LEVEL)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_DEFAULT_LEVEL)
        formatter = logging.Formatter(_FORMATTER_STRING)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate log output
        logger.propagate = False

    return logger
