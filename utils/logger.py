import logging
import os


def setup_logger(
    name: str = "llm_training",
    log_file: str = "logs/training.log",
    level: int = logging.INFO,
):

    os.makedirs(
        os.path.dirname(log_file),
        exist_ok=True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger