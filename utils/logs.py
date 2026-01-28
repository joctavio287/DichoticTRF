from datetime import datetime
from pathlib import Path
import logging
import psutil
import sys

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages
    based on their severity level.

    Parameters
    ----------
        logging.Formatter: Base formatter class
    
    Returns
    -------
        Colored log message string
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset color
        'BOLD': '\033[1m',        # Bold
        'DIM': '\033[2m'          # Dim
    }
    
    def format(
        self, 
        record
    )-> str:
        """
        Format the log record with colors based on severity level.

        Parameters
        ----------
            record: logging.LogRecord
                The log record to format.
        
        Returns
        -------
            str
            The formatted log message with colors.
        """

        # Get the original message
        original_format = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        bold = self.COLORS['BOLD']
        
        # Format: [TIMESTAMP] [LEVEL] [FILE:LINE] MESSAGE
        colored_format = f"{color}{bold}[{record.levelname}]{reset} {color}{original_format}{reset}"
        
        return colored_format

def setup_logger(
    name: str = "Main",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "detailed_logs",
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with colored console output and optional file logging.

    Parameters
    ----------
        name: str
            Logger name
        level: str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: bool
            If True, logs will be saved to a file
        log_dir: str
            Directory to save log files
        console_output: bool
            If True, logs will be printed to console
    Returns
    -------
        logging.Logger
            Configured logger instance    
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(
    name: str = None
) -> logging.Logger:
    """
    Get a logger by name, or create one with default settings if it doesn't exist.

    Parameters
    ----------
        name: str
            Logger name. If None, uses the calling module's name.
    Returns
    -------
        logging.Logger
            Logger instance
    """
    if name is None:
        # Get the calling module name
        frame = sys._getframe(1)
        name = Path(frame.f_globals.get('__file__', 'unknown')).stem
    
    # Check if logger already exists
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    
    # Create new logger with default settings
    return setup_logger(name)

def set_global_log_level(
    level: str
)-> None:
    """
    Set the logging level for all existing loggers.

    Parameters
    ----------
        level: str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    Returns
    -------
        None
    """
    numeric_level = getattr(logging, level.upper())
    
    # Set for all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
        
        # Update handlers
        for handler in logger.handlers:
            handler.setLevel(numeric_level)

def log_function_call(
    func: callable
)-> callable:
    """
    Decorator to log function calls with parameters and execution time
    Example usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    
    Parameters
    ----------
        func: callable
            The function to be decorated.
    
    Returns
    -------
        callable
            The wrapped function with logging.
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        # Log function entry
        logger.debug(f"🔄 Calling {func_name} with args={args}, kwargs={kwargs}", stacklevel=2)
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"✅ {func_name} completed in {duration:.3f}s", stacklevel=2)
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"❌ {func_name} failed after {duration:.3f}s: {str(e)}", stacklevel=2)
            raise
    
    return wrapper

def log_memory_usage(
    logger: logging.Logger = None
)-> None:
    """
    Log the current memory usage of the process in MB.
    Returns
    -------
        None
    """
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if logger is None:
            logger = get_logger()
        logger.debug(f"Memory usage: {memory_mb:.1f} MB", stacklevel=2)
        
    except ImportError:
        pass  # psutil not available

def log_progress(
    current: int, 
    total: int, 
    message: str = "Progress",
    logger: logging.Logger = None,
    start_time: datetime = None,
    level: str = "INFO"
) -> None:
    """
    Log progress of a task with a message and optional ETA.
    Parameters
    ----------
        current: int
            Current progress value
        total: int
            Total value for completion
        message: str
            Message to display with progress
        start_time: datetime, optional
            Start time of the process (for ETA calculation)
    Returns
    -------
        None
    """
    if logger is None:
        logger = get_logger()
    percentage = (current / total) * 100

    eta_str = ""
    if start_time is not None and current > 0:
        elapsed = datetime.now() - start_time
        estimated_total = elapsed / current * total
        remaining = estimated_total - elapsed
        eta_str = f", ETA: {str(remaining).split('.')[0]}"
    
    log_stage(f"{message}: {current}/{total} ({percentage:.1f}%)" + eta_str, logger=logger, level=level, stacklevel=3)

def log_stage(
    message: str,
    level: str = "INFO",
    logger: logging.Logger = None,
    stacklevel: int = 2
)-> None:
    """
    Log a stage header/message using the configured logger.

    Parameters
    ----------
        message: str
            Message to log
        level: str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger: logging.Logger
            Optional logger instance. If None, uses get_logger().
    """
    if logger is None:
        logger = get_logger()
    log_fn = getattr(logger, level.lower(), logger.info)
    cleaned = " ".join(str(message).strip().split())
    log_fn(f"⟫ {cleaned}", stacklevel=stacklevel)

def get_logger_file_paths(
    logger: logging.Logger
) -> list[Path]:
    """
    Return file paths for FileHandlers attached to logger.
    """
    paths: list[Path] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and hasattr(handler, "baseFilename"):
            paths.append(Path(handler.baseFilename))
    return paths

def log_if_false(
    condition: bool,
    message: str,
    logger: logging.Logger,
    level: str = "WARNING",
) -> None:
    """
    Log a message if the given condition is False.

    Parameters
    ----------
    condition: bool
        Condition to evaluate
    message: str
        Message to log if condition is False
    logger: logging.Logger
        Logger instance to use
    level: str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    Returns
    -------
    None
    """
    if not condition:
        log_stage(
            message, 
            level=level, 
            logger=logger,
            stacklevel=3
        )
    else:
        pass