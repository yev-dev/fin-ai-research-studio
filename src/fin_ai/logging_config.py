"""Global logging configuration for FinAI.

Configure a console handler and optional rotating file handler under LOG_DIR.
Importing this module and calling `configure_logging()` at app startup will
ensure module-level `logger.info` calls are visible during Streamlit runs.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import json
import contextvars

# Compute a sensible default log directory relative to the project root
# without importing package config to avoid circular imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


# Context variable holding optional structured log context (e.g., request_id)
_log_context: contextvars.ContextVar[dict] = contextvars.ContextVar("_log_context", default={})


def set_log_context(**kwargs) -> None:
    """Set structured context fields to attach to subsequent log records.

    Example: `set_log_context(request_id="abc", agent="Research_Analyst")`
    """
    ctx = dict(_log_context.get())
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    _log_context.set(ctx)


def clear_log_context() -> None:
    """Clear any structured log context."""
    _log_context.set({})


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter that includes contextvars from `set_log_context`.

    Keeps log record attributes and merges in the structured context.
    """
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach context fields if present
        try:
            ctx = _log_context.get()
            if ctx:
                payload.update(ctx)
        except Exception:
            pass

        # Include exception info if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Include any extra attributes passed via `extra` in logging calls
        standard_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        for k, v in record.__dict__.items():
            if k in standard_keys:
                continue
            if k.startswith("_"):
                continue
            try:
                payload[k] = v
            except Exception:
                try:
                    payload[k] = str(v)
                except Exception:
                    payload[k] = None

        return json.dumps(payload, ensure_ascii=False)



def configure_logging(
    level: int = logging.INFO,
    enable_file: bool = True,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = True,
) -> None:
    """Configure root logging for the application.

    Parameters
    - level: logging level (default INFO)
    - enable_file: whether to enable rotating file logging into `LOG_DIR`
    - log_dir: optional override for the directory to write logs
    - max_bytes, backup_count: rotation settings
    """
    root = logging.getLogger()
    # Avoid adding multiple handlers on repeated calls
    if root.handlers:
        # Update level and return quickly
        root.setLevel(level)
        return

    root.setLevel(level)

    datefmt = "%Y-%m-%d %H:%M:%S"
    if json_format:
        formatter = JsonFormatter(datefmt=datefmt)
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s", datefmt
        )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if enable_file:
        target = Path(log_dir) if log_dir else Path(_DEFAULT_LOG_DIR)
        try:
            target.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                target / "fin_ai.log", maxBytes=max_bytes, backupCount=backup_count
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception:
            # If file handler cannot be created, continue with console only
            root.warning("Failed to create log file handler at %s", str(target))


__all__ = ["configure_logging"]
