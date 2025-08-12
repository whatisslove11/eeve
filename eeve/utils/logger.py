def get_logger():
    from logging import getLogger, INFO, StreamHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger