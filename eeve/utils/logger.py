import os
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

from eeve.files import LOGS_PATH


def get_logger(
    *,
    logger_name: str = "logger",
    logging_dir: str | Path = LOGS_PATH,
    filename: str | None = None,
):
    logger = getLogger(logger_name)
    logger.setLevel(INFO)

    if filename:
        logger.propagate = False

    if not any(type(h) is StreamHandler for h in logger.handlers):
        handler = StreamHandler()
        handler.setFormatter(Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)

    if filename:
        os.makedirs(logging_dir, exist_ok=True)
        log_path = os.path.join(logging_dir, filename)
        abs_path = os.path.abspath(log_path)

        if not any(
            isinstance(h, FileHandler) and h.baseFilename == abs_path
            for h in logger.handlers
        ):
            handler = FileHandler(log_path, mode="a", encoding="utf-8", delay=True)
            handler.setFormatter(Formatter("%(message)s"))
            logger.addHandler(handler)

    return logger
