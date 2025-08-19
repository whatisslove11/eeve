import os
from logging import getLogger, INFO, FileHandler, StreamHandler, Formatter
from eeve.files import LOGS_PATH


def get_logger():
    logger = getLogger(__name__)
    
    if logger.handlers:
        return logger 
    
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    return logger


def get_logs_writer_logger(logging_dir=LOGS_PATH, filename='logs.log'):
    os.makedirs(logging_dir, exist_ok=True)
    log_path = os.path.join(logging_dir, filename)

    logger = getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(INFO)
    logger.propagate = False

    handler = FileHandler(log_path, mode="a", encoding="utf-8", delay=True)
    handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger