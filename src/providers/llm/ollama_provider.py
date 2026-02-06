"""
Ollama Provider Implementation.

Connects to a local Ollama server for running models like Llama, Mistral, etc.
Useful as a fallback or for development without GPU.
"""
import time
import logging
from typing import List, Optional, AsyncIterator
import json

import httpx

from src.providers.llm.base import (
    BaseLLMProvider,
    Message,
    GenerationConfig,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM inference.
    
    Ollama makes it easy to run models locally with simple API.
    Good for development and testing without dedicated GPU infrastructure.
    """
    
    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3.1:8b", "mistral", "qwen2.5")
            api_url: Ollama server URL (default: localhost:11434)
            timeout: Request timeout in seconds
        """
        super().__init__(model, **kwargs)
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        )
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using Ollama's chat API.
        """
        config = config or GenerationConfig()
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": False,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repetition_penalty,
            }
        }
        
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences
        
        try:
            response = await self._client.post(
                f"{self.api_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=data["message"]["content"],
                model=data.get("model", self.model),
                finish_reason="stop" if data.get("done") else None,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                latency_ms=latency_ms,
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama connection error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token.
        """
        config = config or GenerationConfig()
        
        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repetition_penalty,
            }
        }
        
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.api_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama stream error: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama stream connection error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = await self._client.get(f"{self.api_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = await self._client.get(f"{self.api_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            async with self._client.stream(
                "POST",
                f"{self.api_url}/api/pull",
                json={"name": model},
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        logger.info(f"Pulling {model}: {status}")
                        if "error" in data:
                            logger.error(f"Pull error: {data['error']}")
                            return False
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
