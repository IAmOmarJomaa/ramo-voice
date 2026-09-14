"""
ramo_common
===========
Shared utilities, observability, and logging infrastructure for ramO Engine.
"""

from .logging import (
    setup_service_logging,
    tail_service_log,
    close_service_logging,
    LogStreamHub,
    global_log_hub,
)

__all__ = [
    "setup_service_logging",
    "tail_service_log",
    "close_service_logging",
    "LogStreamHub",
    "global_log_hub",
]
