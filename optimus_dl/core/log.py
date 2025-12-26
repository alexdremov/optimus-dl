import logging
import sys
import warnings
from pathlib import Path

import colorlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    use_rich: bool = True,
    use_colors: bool = True,
    format_string: str | None = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up beautiful Python logging with colors and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        use_rich: Use Rich library for enhanced console output
        use_colors: Enable colored output (ignored if use_rich=True)
        format_string: Custom format string (uses default if None)
        date_format: Date format for timestamps

    Returns:
        Configured logger instance

    Example:
        ```python
        logger = setup_logging(level="DEBUG", log_file="app.log")
        logger.info("Application started")
        logger.error("Something went wrong!")
        ```
    """
    warnings.filterwarnings(
        "ignore",
        message=".*Subclassing `Dict` in Structured Config classes is deprecated.*",
    )

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set the logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)

    handlers = []

    # Console handler setup
    if use_rich:
        # Rich handler for beautiful console output
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(level)
        handlers.append(rich_handler)

    else:
        # Standard console handler with optional colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if use_colors:
            # Colorlog for colored output
            color_formatter = colorlog.ColoredFormatter(
                format_string
                or "%(log_color)s%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                datefmt=date_format,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
            console_handler.setFormatter(color_formatter)
        else:
            # Standard formatter
            formatter = logging.Formatter(
                format_string
                or "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                datefmt=date_format,
            )
            console_handler.setFormatter(formatter)

        handlers.append(console_handler)

    # File handler setup (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        # File output doesn't need colors
        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt=date_format,
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Add all handlers to the root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Return a logger instance
    return logging.getLogger(__name__)


def warn_once(logger: logging.Logger, message: str) -> None:
    """Log a warning message only once, even if called multiple times.

    This is useful for deprecation warnings or other messages that should
    only appear once per program execution, even if the code path is
    executed multiple times.

    Args:
        logger: Logger instance to log the warning to.
        message: Warning message to log (only logged once).

    Example:
        ```python
        logger = logging.getLogger(__name__)
        warn_once(logger, "This feature is deprecated")
        warn_once(logger, "This feature is deprecated")  # Won't log again
        ```
    """
    if not hasattr(warn_once, "logged_messages"):
        warn_once.logged_messages = set()
    if message not in warn_once.logged_messages:
        logger.warning(message)
        warn_once.logged_messages.add(message)
