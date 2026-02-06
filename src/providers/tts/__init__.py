"""
Text-to-Speech (TTS) providers.

Provides speech synthesis capabilities for the AI Interviewer.
"""
from src.providers.tts.pyttsx3_provider import (
    Pyttsx3TTSProvider,
    VoiceGender,
    VoiceInfo,
    SynthesisResult,
    get_tts_provider,
    get_tts_provider_async,
)

__all__ = [
    "Pyttsx3TTSProvider",
    "VoiceGender",
    "VoiceInfo",
    "SynthesisResult",
    "get_tts_provider",
    "get_tts_provider_async",
]
