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


def test_safe_stream_handler_handles_emojis_on_cp1252():
    from ramo_common.logging import SafeStreamHandler
    import logging

    class StrictCP1252Stream:
        def __init__(self):
            self.encoding = "cp1252"
            self.written = []

        def write(self, s: str):
            # Strict cp1252 check - raises UnicodeEncodeError on emojis
            s.encode("cp1252", errors="strict")
            self.written.append(s)

        def flush(self):
            pass

    mock_stream = StrictCP1252Stream()
    handler = SafeStreamHandler(mock_stream)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="🟢 [WS Connected] Session sess_112723 registered. 👥 🔊",
        args=(),
        exc_info=None,
    )

    # Should not raise UnicodeEncodeError or call handleError
    handler.emit(record)
    assert len(mock_stream.written) == 1
    assert "[WS Connected]" in mock_stream.written[0]
