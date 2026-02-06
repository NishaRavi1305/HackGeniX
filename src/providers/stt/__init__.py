"""
Speech-to-Text (STT) providers.

Provides speech recognition capabilities for the AI Interviewer.
Supports both original Whisper and faster-whisper implementations.
"""
from src.providers.stt.whisper_provider import (
    WhisperSTTProvider,
    WhisperModel,
    TranscriptionResult,
    TranscriptionSegment,
    get_whisper_provider,
    get_whisper_provider_async,
)
from src.providers.stt.faster_whisper_provider import (
    FasterWhisperSTTProvider,
    get_faster_whisper_provider,
    get_faster_whisper_provider_async,
)

__all__ = [
    # Original Whisper
    "WhisperSTTProvider",
    "WhisperModel",
    "TranscriptionResult",
    "TranscriptionSegment",
    "get_whisper_provider",
    "get_whisper_provider_async",
    # Faster Whisper (recommended)
    "FasterWhisperSTTProvider",
    "get_faster_whisper_provider",
    "get_faster_whisper_provider_async",
]
