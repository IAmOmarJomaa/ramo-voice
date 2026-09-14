"""
services.common.tests.test_logging
==================================
Tests for structured rotating file logger, log tailing, and live streaming hub.
"""

import os
import tempfile
import pytest
import asyncio


def test_setup_service_logging_creates_file_and_rotates():
    from ramo_common.logging import setup_service_logging, tail_service_log, close_service_logging

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = setup_service_logging("test_service", log_dir=tmp_dir, max_bytes=1024, backup_count=2)
        
        # Log a few lines
        for i in range(50):
            logger.info(f"Log event #{i} - test line for checking file creation")

        log_file = os.path.join(tmp_dir, "test_service.log")
        assert os.path.exists(log_file), "Log file was not created"

        # Verify tailing
        lines = tail_service_log("test_service", n=10, log_dir=tmp_dir)
        assert len(lines) <= 10
        assert len(lines) > 0
        assert "Log event #49" in lines[-1]

        # Cleanly release handlers so temporary directory can be deleted on Windows
        close_service_logging(logger)


def test_tail_service_log_handles_missing_file():
    from ramo_common.logging import tail_service_log

    with tempfile.TemporaryDirectory() as tmp_dir:
        lines = tail_service_log("nonexistent_service", n=20, log_dir=tmp_dir)
        assert lines == []


@pytest.mark.asyncio
async def test_log_stream_hub_pub_sub():
    from ramo_common.logging import LogStreamHub

    hub = LogStreamHub()
    queue = hub.subscribe()

    await hub.publish("test_service", "INFO", "Hello live log subscriber!")

    assert not queue.empty()
    item = await queue.get()
    assert item["service"] == "test_service"
    assert item["level"] == "INFO"
    assert "Hello live log subscriber!" in item["message"]

    hub.unsubscribe(queue)
