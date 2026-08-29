import logging


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=(
            "%(asctime)s %(levelname)s "
            "event=%(message)s "
            "logger=%(name)s"
        ),
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
