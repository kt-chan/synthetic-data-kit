import logging
import logging.handlers
import os
import threading
import inspect
from functools import wraps
from typing import Optional, Callable, Any
from pathlib import Path

class AppLogger:
    """
    Thread-safe logging class for Flash applications that supports multiple modules
    logging to a single file with automatic class name detection and consistent formatting.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppLogger, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 log_file: str = "app.log", 
                 log_level: int = logging.INFO,
                 when: str = 'midnight', 
                 interval: int = 1, 
                 backup_count: int = 7):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to the log file
            log_level: Default logging level
            when: When to rotate logs ('S', 'M', 'H', 'D', 'W', 'midnight')
            interval: Rotation interval
            backup_count: Number of backup files to keep
        """
        # Check if already initialized to avoid reinitialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.log_file = log_file
        self.log_level = log_level
        self.loggers = {}
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create formatter with safe handling for missing classname
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s'
        )
        
        # Create timed rotating file handler
        self.handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when=when, interval=interval, backupCount=backup_count
        )
        self.handler.setFormatter(formatter)
        self.handler.setLevel(log_level)
        
        # Add a filter to safely handle classname attribute
        class SafeClassNameFilter(logging.Filter):
            def filter(self, record):
                # Ensure classname attribute exists to prevent KeyError in formatter
                if not hasattr(record, 'classname'):
                    record.classname = 'Module'
                return True
                
        self.handler.addFilter(SafeClassNameFilter())
        
        # Set up root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(log_level)
        self.root_logger.addHandler(self.handler)
        
        self._initialized = True
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific module with automatic class name detection.
        
        Args:
            module_name: Name of the module requesting the logger
            
        Returns:
            Configured logger instance
        """
        # Use calling module name if not provided
        if module_name is None:
            # Get the name of the calling module
            frame = inspect.currentframe().f_back
            module_name = frame.f_globals['__name__']
        
        if module_name not in self.loggers:
            logger = logging.getLogger(module_name)
            logger.setLevel(self.log_level)
            
            # Add a filter to include class name in log records
            class ClassNameFilter(logging.Filter):
                def filter(self, record):
                    # Try to extract class name from the stack
                    try:
                        frame = inspect.currentframe()
                        # Go up the stack to find the class context
                        for i in range(6):  # Reasonable limit to search
                            frame = frame.f_back
                            if frame is None:
                                break
                            # Check if 'self' exists in the frame's locals
                            if 'self' in frame.f_locals:
                                instance = frame.f_locals['self']
                                record.classname = instance.__class__.__name__
                                break
                        else:
                            record.classname = 'Module'
                    except:
                        record.classname = 'Module'
                    return True
            
            logger.addFilter(ClassNameFilter())
            self.loggers[module_name] = logger
        
        return self.loggers[module_name]
    
    def set_level(self, level: int) -> None:
        """
        Set the logging level for all loggers.
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self.log_level = level
        self.handler.setLevel(level)
        for logger in self.loggers.values():
            logger.setLevel(level)
    
    def shutdown(self) -> None:
        """Properly shutdown the logging system."""
        logging.shutdown()

# Global logger instance
_flash_logger = None

def setup_logging(log_file: str = "app.log", 
                 log_level: int = logging.INFO,
                 **kwargs) -> None:
    """
    Initialize the logging system.
    
    Args:
        log_file: Path to the log file
        log_level: Default logging level
        **kwargs: Additional arguments for TimedRotatingFileHandler
    """
    global _flash_logger
    _flash_logger = AppLogger(log_file, log_level, **kwargs)

def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger
        
    Returns:
        Configured logger instance
    """
    if _flash_logger is None:
        # Initialize with default values if not already set up
        setup_logging()
    return _flash_logger.get_logger(module_name)

def log_function_call(level: int = logging.DEBUG) -> Callable:
    """
    Decorator to log function calls with parameters and return values.
    
    Args:
        level: Logging level for the function call information
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger()
            
            # Get class name if method
            class_name = 'Module'
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
            
            # Log function entry
            logger.log(level, f"Entering {class_name}.{func.__name__} with args: {args}, kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                # Log function exit
                logger.log(level, f"Exiting {class_name}.{func.__name__} with result: {result}")
                return result
            except Exception as e:
                # Log exception
                logger.error(f"Exception in {class_name}.{func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# # Example usage with Path conversion
# if __name__ == "__main__":
#     # Define the default log directory
#     DEFAULT_LOG_DIR = Path(__file__).parent / "logs"
    
#     # Setup logging with the path conversion
#     setup_logging(
#         log_file=str((DEFAULT_LOG_DIR / "app.log").resolve()),
#         log_level=logging.DEBUG, 
#         when='midnight', 
#         backup_count=30
#     )
    
#     # Example module 1
#     class DatabaseModule:
#         def __init__(self):
#             self.logger = get_logger(__name__)
            
#         @log_function_call()
#         def connect(self, database_url):
#             self.logger.info(f"Connecting to database: {database_url}")
#             # Simulate database connection
#             return {"status": "connected", "url": database_url}
    
#     # Example module 2
#     class APIModule:
#         def __init__(self):
#             self.logger = get_logger(__name__)
            
#         @log_function_call(logging.INFO)
#         def process_request(self, request_data):
#             self.logger.info(f"Processing request: {request_data}")
#             # Simulate processing
#             if "error" in request_data:
#                 raise ValueError("Invalid request data")
#             return {"status": "processed", "data": request_data}
    
#     # Test the logging
#     db_module = DatabaseModule()
#     api_module = APIModule()
    
#     # Successful operations
#     db_result = db_module.connect("postgresql://localhost/mydb")
#     api_result = api_module.process_request({"action": "get_data"})
    
#     # Operation with error
#     try:
#         api_module.process_request({"action": "error"})
#     except ValueError:
#         pass  # Expected error
    
#     # Shutdown logging
#     if _flash_logger:
#         _flash_logger.shutdown()