"""
-------------------------------------------------------
Logger Module for Soccer Tracker Project
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
 -------------------------------------------------------
"""

# Imports
import logging

# Constants
logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.WARNING,
)
log_levels = {
    "info": logging.INFO,
    "warn": logging.WARN,
    "debug": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(name, loglevel):
    """
    -------------------------------------------------------
    Returns a logger object
    -------------------------------------------------------
    Parameters:
        name - The Name of the Module (str)
        loglevel - the log level of the assigned logger (str)
    Returns:
        logger - Python Logger Object
    -------------------------------------------------------
    """
    lvl = loglevel.lower()
    assert lvl in (
        "info",
        "warn",
        "debug",
        "error",
        "critical",
    ), "Unsupported log level, choose from info, warn, debug, error, critical"
    logger = logging.getLogger(name=name)
    logger.setLevel(
        log_levels.get(loglevel.lower(), logging.NOTSET)
    )  # Lowercase for case-insensitivity
    return logger
