import logging

BLUE = "\033[94m"
RESET = "\033[0m"

def create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter(BLUE+'%(name)s - %(levelname)s - %(message)s'+RESET)
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger
