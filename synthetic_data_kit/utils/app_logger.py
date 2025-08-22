import logging
import queue
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional


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
            queues_copy = list(self.queues)
            for q in queues_copy:
                try:
                    q.put(message)
                except:
                    self.queues.discard(q)


class SSEHandler(logging.Handler):
    def __init__(self, queue_manager, maxsize=1000):
        super().__init__()
        self.queue_manager = queue_manager
        self.maxsize = maxsize

    def emit(self, record):
        try:
            formatted_message = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ).format(record)
            log_queue = queue.Queue(maxsize=self.maxsize)
            try:
                log_queue.put_nowait(formatted_message)
            except queue.Full:
                logging.error("Log queue overflow, dropping message")

            self.queue_manager.broadcast_message(formatted_message)
        except Exception:
            self.handleError(record)


_sse_queue_manager = None


def get_sse_queue_manager():
    global _sse_queue_manager
    if _sse_queue_manager is None:
        _sse_queue_manager = SSEQueueManager()
    return _sse_queue_manager


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_sse: bool = False,
    enable_rotation: bool = False,
    max_bytes: int = 10000000,
    backup_count: int = 7,
) -> logging.Logger:
    sse_queue_manager = get_sse_queue_manager() if enable_sse else None

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        if enable_rotation:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, when="midnight"
            )
        else:
            file_handler = logging.FileHandler(log_file)

        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if enable_sse and sse_queue_manager:
        sse_handler = SSEHandler(sse_queue_manager)
        sse_handler.setFormatter(formatter)
        root_logger.addHandler(sse_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def add_sse_queue(log_queue: queue.Queue) -> None:
    sse_queue_manager = get_sse_queue_manager()
    sse_queue_manager.add_queue(log_queue)


def remove_sse_queue(log_queue: queue.Queue) -> None:
    sse_queue_manager = get_sse_queue_manager()
    sse_queue_manager.remove_queue(log_queue)


def log_function_call(logger: logging.Logger):
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
