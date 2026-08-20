"""fin_ai package.

On import, configure application logging if not already configured. This
ensures `logger.info` calls across modules show up during Streamlit runs.
"""
from fin_ai.logging_config import configure_logging, set_log_context, clear_log_context

# Configure default logging: INFO level, enable file logging into LOG_DIR,
# and output JSON-structured logs by default for better traceability.
configure_logging(level=20, enable_file=True, json_format=True)

# Expose context helpers at package level for convenience
__all__ = ["configure_logging", "set_log_context", "clear_log_context"]
