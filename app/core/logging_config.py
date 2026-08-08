import logging
from logging.handlers import RotatingFileHandler
import sys
from app.core.config import get_settings
from app.core.request_context import get_request_id

settings = get_settings()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("knowledge_base")
    logger.setLevel(settings.log_level.upper())

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    request_id_filter = RequestIdFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3,
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_id_filter)
    logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    return logging.getLogger("knowledge_base").getChild(module_name)