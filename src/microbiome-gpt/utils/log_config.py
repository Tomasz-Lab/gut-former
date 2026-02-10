import logging

LOG_FORMAT = "%(message)s"

def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT
    )