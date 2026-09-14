import logging
import sys
import json
from datetime import datetime
from pathlib import Path


# Attribute names a stock logging.LogRecord always carries. A caller's
# `logger.info(msg, extra={...})` lands its entries directly on the
# record as individual attributes (logging.Logger.makeRecord uses
# setattr, it does not nest them under a `.extra` attribute) -- so
# `hasattr(record, "extra")` below was always False and every `extra=`
# a caller ever passed was silently dropped from the JSON log output.
_STANDARD_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys())


class CustomJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        extra_attrs = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS
        }
        if extra_attrs:
            log_data.update(extra_attrs)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # default=str: a log call must never raise for having passed a
        # non-JSON-serializable value in `extra`.
        return json.dumps(log_data, default=str)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (JSON).
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(CustomJSONFormatter())
    root_logger.addHandler(file_handler)

    # Error file handler.
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(CustomJSONFormatter())
    root_logger.addHandler(error_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


logger = setup_logging()
