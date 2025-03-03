import logging


class LoggerFactory:

    @staticmethod
    def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger
