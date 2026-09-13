"""
ramo_gateway.server
===================
FastAPI duplex WebSocket server connecting client microphones to STT, LLM, and TTS microservices.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .state_machine import ConversationStateMachine, SessionState

logger = logging.getLogger("ramo_gateway.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ramo_gateway server...")
    yield


app = FastAPI(
    title="ramO Voice Gateway API",
    version="0.1.0",
    description="Sovereign Duplex Real-Time Voice Gateway with Barge-in",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ramo-gateway",
        "vad_state": "active"
    }


@app.websocket("/v1/realtime")
async def realtime_session(ws: WebSocket):
    """
    Real-time duplex conversational WebSocket.
    """
    await ws.accept()
    sm = ConversationStateMachine(silence_timeout_sec=0.3)

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                # Incoming audio
                interrupted = sm.on_speech_start()
                if interrupted:
                    # User interrupted playback: broadcast interrupt signal
                    await ws.send_json({"type": "interrupt"})

            elif "text" in msg and msg["text"]:
                import json
                try:
                    data = json.loads(msg["text"])
                    event_type = data.get("type")
                    if event_type == "silence":
                        sm.on_speech_stop()
                        await ws.send_json({"type": "state_change", "state": sm.state.value})
                    elif event_type == "playback_start":
                        sm.set_state(SessionState.PLAYING)
                    elif event_type == "playback_end":
                        sm.set_state(SessionState.LISTENING)
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Gateway client disconnected.")
