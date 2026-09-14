"""
ramo_common.logging
===================
Enterprise-grade rotating file logger, structured formatting, log tailing,
and live async WebSocket log streaming hub for ramO Engine microservices.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Set, Optional
import asyncio
from datetime import datetime, timezone


class LogStreamHub:
    """
    In-memory asynchronous publication/subscription hub for streaming live logs
    to WebSocket clients and telemetry monitors.
    """

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unregister an existing subscriber queue."""
        self._subscribers.discard(queue)

    async def publish(self, service: str, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Publish a log event to all active subscriber queues."""
        if not self._subscribers:
            return

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "level": level,
            "message": message,
            "extra": extra or {},
        }

        # Dispatch non-blockingly to all queues
        for q in list(self._subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(payload)
            except Exception:
                pass


# Global singleton hub
global_log_hub = LogStreamHub()


class HubHandler(logging.Handler):
    """Logging handler that emits structured records to the async LogStreamHub."""

    def __init__(self, service_name: str, hub: LogStreamHub):
        super().__init__()
        self.service_name = service_name
        self.hub = hub

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Try scheduling publication on active event loop if available
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.hub.publish(
                        service=self.service_name,
                        level=record.levelname,
                        message=msg,
                    )
                )
            except RuntimeError:
                # No active running loop in current thread
                pass
        except Exception:
            self.handleError(record)


def setup_service_logging(
    service_name: str,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per log file
    backup_count: int = 3,               # 3 backups (logs.1, logs.2, logs.3)
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures standard structured rotating file logging for a microservice.
    Writes to both standard output and `logs/<service_name>.log`.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{service_name}.log")

    logger = logging.getLogger(service_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if re-initialized
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path) for h in logger.handlers):
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Rotating File Handler
        rfh = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        rfh.setFormatter(formatter)
        rfh.setLevel(level)
        logger.addHandler(rfh)

        # 2. Console Handler (if not present)
        has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not has_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level)
            logger.addHandler(ch)

        # 3. Hub Stream Handler
        hub_h = HubHandler(service_name=service_name, hub=global_log_hub)
        hub_h.setFormatter(formatter)
        hub_h.setLevel(level)
        logger.addHandler(hub_h)

    return logger


def tail_service_log(service_name: str, n: int = 100, log_dir: str = "logs") -> List[str]:
    """
    Returns the last `n` lines from the service's log file.
    Safe against missing files or locked files.
    """
    log_path = os.path.join(log_dir, f"{service_name}.log")
    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.rstrip("\r\n") for line in lines[-n:]]
    except Exception as e:
        return [f"[ERROR reading logs]: {e}"]


def close_service_logging(logger: logging.Logger) -> None:
    """Closes and removes all handlers from a logger to release file locks."""
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
