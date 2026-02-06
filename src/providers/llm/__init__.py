"""
LLM Providers Package.

Provides plug-and-play LLM backends including vLLM, Ollama, and more.
"""
from src.providers.llm.base import (
    BaseLLMProvider,
    LLMProvider,
    Message,
    GenerationConfig,
    LLMResponse,
    system_message,
    user_message,
    assistant_message,
)
from src.providers.llm.vllm_provider import VLLMProvider
from src.providers.llm.ollama_provider import OllamaProvider
from src.providers.llm.factory import (
    LLMProviderFactory,
    get_llm_provider,
    get_llm_provider_sync,
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMProvider",
    "Message",
    "GenerationConfig",
    "LLMResponse",
    # Message helpers
    "system_message",
    "user_message",
    "assistant_message",
    # Providers
    "VLLMProvider",
    "OllamaProvider",
    # Factory
    "LLMProviderFactory",
    "get_llm_provider",
    "get_llm_provider_sync",
]
