import pytest
from unittest.mock import AsyncMock, MagicMock
from ramo_translate.router import TranslationRouter, TranslationResult
from ramo_translate.context_tracker import MeetingContextTracker


@pytest.mark.asyncio
async def test_router_same_language_short_circuit():
    router = TranslationRouter()
    result = await router.translate(
        text="Hello world",
        source_lang="en",
        target_lang="en",
    )
    assert result.translated_text == "Hello world"
    assert result.is_bypass


@pytest.mark.asyncio
async def test_router_fast_bypass_hit():
    router = TranslationRouter()
    result = await router.translate(
        text="Okay.",
        source_lang="en",
        target_lang="fr",
    )
    assert result.translated_text == "D'accord."
    assert result.is_bypass


@pytest.mark.asyncio
async def test_router_engine_dispatch_with_context():
    mock_engine = AsyncMock()
    mock_engine.generate_translation.return_value = "Bonjour tout le monde."

    router = TranslationRouter(engine=mock_engine)
    result = await router.translate(
        text="Hello everyone.",
        source_lang="en",
        target_lang="fr",
        session_id="sync_1",
        speaker_id="SPEAKER_00",
    )

    assert result.translated_text == "Bonjour tout le monde."
    assert not result.is_bypass
    assert mock_engine.generate_translation.called
