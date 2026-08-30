"""
Structured logging configuration module.

Provides JSON-formatted logging for production environments and
human-readable formatting for development. Enables:
- Structured logging with context fields
- Log level configuration
- Log aggregation compatibility (ELK, Loki, CloudWatch)
- Performance tracking

Author: Instrument Recognizer Team
Date: 2024
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


# ============================================================================
# LOGGING CONFIGURATION CONSTANTS
# ============================================================================


class LogConfig:
    """Configuration constants for logging system."""
    
    # Log levels
    LEVEL_DEBUG = logging.DEBUG          # 10
    LEVEL_INFO = logging.INFO            # 20
    LEVEL_WARNING = logging.WARNING       # 30
    LEVEL_ERROR = logging.ERROR           # 40
    LEVEL_CRITICAL = logging.CRITICAL     # 50
    
    # Default log level from environment or INFO
    DEFAULT_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Log directory
    LOG_DIR = Path('logs')
    LOG_DIR.mkdir(exist_ok=True)
    
    # Log files
    LOG_FILE = LOG_DIR / 'app.log'
    ERROR_LOG_FILE = LOG_DIR / 'errors.log'
    
    # Log format templates
    JSON_FORMAT = '%(message)s'
    CONSOLE_FORMAT = (
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ============================================================================
# CUSTOM FORMATTERS
# ============================================================================


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON for structured logging.
    
    Benefits:
    - Machine-parseable format
    - Compatible with log aggregation services (ELK, Loki)
    - Enables complex queries and analysis
    - Preserves context fields
    
    Example output:
        {
            "timestamp": "2024-01-15T10:30:45.123Z",
            "level": "INFO",
            "logger": "app.api",
            "message": "Analysis request received",
            "request_id": "req_abc123",
            "user_id": "user_456"
        }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: LogRecord instance from logging framework
            
        Returns:
            JSON-formatted string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """
    Custom formatter for console output with colors for development.
    
    Provides human-readable output with timestamps and color coding
    by log level for easier debugging during development.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors for console output.
        
        Args:
            record: LogRecord instance
            
        Returns:
            Formatted string with ANSI color codes
        """
        levelname = record.levelname
        color = self.COLORS.get(levelname, '')
        reset = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{levelname}{reset}"
        
        return super().format(record)


# ============================================================================
# LOGGER SETUP FUNCTIONS
# ============================================================================


def setup_logger(
    name: str,
    level: str = LogConfig.DEFAULT_LEVEL,
    enable_file_logging: bool = True,
    enable_console: bool = True
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Sets up:
    - Console handler (human-readable)
    - File handler (JSON format for aggregation)
    - Error file handler (errors only)
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        enable_file_logging: Whether to write logs to files
        enable_console: Whether to output to console
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started", extra={'version': '2.0.0'})
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # Console Handler (human-readable)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ConsoleFormatter(LogConfig.CONSOLE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File Handler (JSON format)
    if enable_file_logging:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=LogConfig.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = JSONFormatter(LogConfig.JSON_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Error File Handler (errors only)
    if enable_file_logging:
        try:
            error_handler = logging.handlers.RotatingFileHandler(
                filename=LogConfig.ERROR_LOG_FILE,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = JSONFormatter(LogConfig.JSON_FORMAT)
            error_handler.setFormatter(error_formatter)
            logger.addHandler(error_handler)
        except Exception as e:
            print(f"Warning: Could not setup error file logging: {e}")
    
    return logger


# ============================================================================
# LOGGING HELPER FUNCTIONS
# ============================================================================


def log_context_wrapper(logger: logging.Logger) -> 'LogContextManager':
    """
    Create a context manager for adding contextual fields to logs.
    
    Args:
        logger: Logger instance
        
    Returns:
        LogContextManager instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> with log_context_wrapper(logger).add_context(request_id='abc123'):
        ...     logger.info("Processing request")  # Will include request_id
    """
    return LogContextManager(logger)


class LogContextManager:
    """
    Context manager for adding contextual fields to log records.
    
    Allows adding structured context that persists across multiple
    log statements within a scope.
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> ctx = LogContextManager(logger)
        >>> with ctx.add_context(user_id='user_123', action='login'):
        ...     logger.info("User action")  # Includes user_id and action
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize context manager."""
        self.logger = logger
        self.context: Dict[str, Any] = {}
    
    def add_context(self, **kwargs) -> 'LogContextManager':
        """
        Add context fields to this logger.
        
        Args:
            **kwargs: Key-value pairs to add to log context
            
        Returns:
            Self for context manager usage
        """
        self.context.update(kwargs)
        return self
    
    def __enter__(self) -> 'LogContextManager':
        """Enter context (for 'with' statement)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and clear context fields."""
        self.context.clear()
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self._log(self.logger.info, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        self._log(self.logger.error, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self._log(self.logger.warning, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self._log(self.logger.debug, message, **kwargs)
    
    def _log(self, log_func, message: str, **kwargs) -> None:
        """
        Internal helper to log with context.
        
        Args:
            log_func: Logging function (logger.info, logger.error, etc.)
            message: Message to log
            **kwargs: Additional context fields
        """
        combined_context = {**self.context, **kwargs}
        log_func(message, extra={'extra_fields': combined_context})


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


# Create module-level logger
logger = setup_logger(__name__)
logger.debug(f"Logging system initialized (level: {LogConfig.DEFAULT_LEVEL})")
