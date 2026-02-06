"""
vLLM Provider Implementation.

Connects to a vLLM server running Llama-3.1-8B or other models
via OpenAI-compatible API.
"""
import time
import logging
from typing import List, Optional, AsyncIterator

import httpx

from src.providers.llm.base import (
    BaseLLMProvider,
    Message,
    GenerationConfig,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """
    vLLM provider using OpenAI-compatible API.
    
    vLLM serves models with high throughput and low latency.
    Supports Llama, Mistral, Qwen, and other models.
    """
    
    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:8001/v1",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Initialize vLLM provider.
        
        Args:
            model: Model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            api_url: vLLM server URL (default: localhost:8001)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        super().__init__(model, **kwargs)
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # HTTP client with connection pooling
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using vLLM's chat completions API.
        """
        config = config or GenerationConfig()
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": False,
        }
        
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        
        try:
            response = await self._client.post(
                f"{self.api_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage"),
                latency_ms=latency_ms,
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"vLLM connection error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token using SSE.
        """
        config = config or GenerationConfig()
        
        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.api_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        import json
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM stream error: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"vLLM stream connection error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self._client.get(f"{self.api_url}/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the vLLM server."""
        try:
            response = await self._client.get(f"{self.api_url}/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
