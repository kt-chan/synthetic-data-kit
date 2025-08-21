import json
import logging
import queue
import threading
from typing import Optional

# Create a queue manager for SSE log messages
class SSEQueueManager:
    def __init__(self):
        self.queues = set()
        self.lock = threading.Lock()
    
    def add_queue(self, q):
        with self.lock:
            self.queues.add(q)
    
    def remove_queue(self, q):
        with self.lock:
            if q in self.queues:
                self.queues.remove(q)
    
    def broadcast_message(self, message):
        with self.lock:
            for q in list(self.queues):  # Create a copy to avoid issues during iteration
                try:
                    q.put(message)
                except:
                    # Remove queue if it's no longer valid
                    self.queues.discard(q)

# Custom logging handler that sends messages to SSE clients
class SSEHandler(logging.Handler):
    def __init__(self, queue_manager):
        super().__init__()
        self.queue_manager = queue_manager
    
    def emit(self, record):
        try:
            # Format the log record in the standard format
            formatted_message = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s').format(record)
            
            # Create JSON with the formatted message
            log_data = formatted_message
            msg = json.dumps(log_data)
            self.queue_manager.broadcast_message(msg)
        except Exception:
            self.handleError(record)

# Global SSE queue manager instance
_sse_queue_manager = None

def get_sse_queue_manager():
    """Get or create the global SSE queue manager"""
    global _sse_queue_manager
    if _sse_queue_manager is None:
        _sse_queue_manager = SSEQueueManager()
    return _sse_queue_manager

def setup_logging(log_file: Optional[str] = None, 
                 log_level: int = logging.INFO, 
                 enable_sse: bool = False) -> logging.Logger:
    """
    Set up logging with SSE capability
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        enable_sse: Whether to enable SSE broadcasting
    
    Returns:
        Root logger instance
    """
    # Create a queue manager for SSE if enabled
    sse_queue_manager = get_sse_queue_manager() if enable_sse else None
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create standard formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add SSE handler if enabled
    if enable_sse and sse_queue_manager:
        sse_handler = SSEHandler(sse_queue_manager)
        # Use the same formatter for consistency
        sse_handler.setFormatter(formatter)
        root_logger.addHandler(sse_handler)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def add_sse_queue(log_queue: queue.Queue) -> None:
    """
    Add a queue to the SSE queue manager for broadcasting
    
    Args:
        log_queue: Queue to add for SSE broadcasting
    """
    sse_queue_manager = get_sse_queue_manager()
    sse_queue_manager.add_queue(log_queue)

def remove_sse_queue(log_queue: queue.Queue) -> None:
    """
    Remove a queue from the SSE queue manager
    
    Args:
        log_queue: Queue to remove from SSE broadcasting
    """
    sse_queue_manager = get_sse_queue_manager()
    sse_queue_manager.remove_queue(log_queue)

# Function decorator for logging function calls
def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and return values
    
    Args:
        logger: Logger instance to use for logging
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised exception: {e}")
                raise
        return wrapper
    return decorator