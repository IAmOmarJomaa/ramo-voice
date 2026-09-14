"""
ramo_translate.server
=====================
FastAPI HTTP REST & WebSocket streaming server for Translation & Meeting Intelligence.
Runs standalone on Port 50053.
"""

import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .router import TranslationRouter
from .action_detector import detect_action_item

logger = logging.getLogger("ramo_translate.server")

app = FastAPI(
    title="ramO Translation & Intelligence API",
    description="Sovereign Real-Time Translation & Meeting Action Detection Microservice",
    version="0.1.0",
)

router = TranslationRouter()


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text utterance to translate.")
    source_language: str = Field(default="en", description="Source language code (e.g. en, fr, es).")
    target_language: str = Field(default="fr", description="Target language code (e.g. fr, es, de, ar).")
    session_id: str = Field(default="default", description="Meeting session ID for context tracking.")
    speaker_id: str = Field(default="default", description="Speaker ID for symbol assignment.")


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    is_bypass: bool
    detected_action: Optional[str] = None
    speaker_symbol: Optional[str] = None


class ActionItemRequest(BaseModel):
    text: str = Field(..., description="Utterance text to analyze for commitments or schedule changes.")
    speaker_id: str = Field(default="default", description="Speaker attribution.")


class ActionItemResponse(BaseModel):
    detected_action: Optional[str]
    speaker_id: str


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ramo-translation",
        "engine": router.engine.engine_id,
        "default_port": 50053,
    }


@app.post("/v1/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    """Translate text with 3-tier meeting context and 0ms fast-bypass."""
    result = await router.translate(
        text=req.text,
        source_lang=req.source_language,
        target_lang=req.target_language,
        session_id=req.session_id,
        speaker_id=req.speaker_id,
    )
    return TranslationResponse(
        translated_text=result.translated_text,
        source_language=result.source_language,
        target_language=result.target_language,
        is_bypass=result.is_bypass,
        detected_action=result.detected_action,
        speaker_symbol=result.speaker_symbol,
    )


@app.post("/v1/meeting/action_items", response_model=ActionItemResponse)
async def extract_action_items(req: ActionItemRequest):
    """Classify meeting action items into task_delegation, schedule_change, or fact_check."""
    action = detect_action_item(req.text)
    return ActionItemResponse(
        detected_action=action,
        speaker_id=req.speaker_id,
    )


@app.websocket("/v1/translate/stream")
async def translate_stream(websocket: WebSocket):
    """WebSocket streaming token delivery for real-time translation."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            source_lang = data.get("source_language", "en")
            target_lang = data.get("target_language", "fr")
            session_id = data.get("session_id", "default")
            speaker_id = data.get("speaker_id", "default")

            res = await router.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                speaker_id=speaker_id,
            )

            await websocket.send_json({
                "translated_text": res.translated_text,
                "is_bypass": res.is_bypass,
                "detected_action": res.detected_action,
                "speaker_symbol": res.speaker_symbol,
            })
    except WebSocketDisconnect:
        logger.info("Client disconnected from translation stream")
