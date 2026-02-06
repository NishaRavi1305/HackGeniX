"""
LLM Provider Unit Tests.

Tests Ollama provider and factory with mocked HTTP responses.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

import httpx

from src.providers.llm.base import (
    Message,
    GenerationConfig,
    LLMResponse,
    system_message,
    user_message,
    assistant_message,
)
from src.providers.llm.ollama_provider import OllamaProvider
from src.providers.llm.factory import LLMProviderFactory, get_llm_provider, get_llm_provider_sync


class TestOllamaProvider:
    """Test OllamaProvider with mocked HTTP."""
    
    @pytest.fixture
    def provider(self):
        """Create OllamaProvider instance."""
        return OllamaProvider(model="qwen2.5:3b", api_url="http://localhost:11434")
    
    @pytest.fixture
    def sample_messages(self):
        """Sample chat messages."""
        return [
            system_message("You are a helpful assistant."),
            user_message("Hello, how are you?"),
        ]
    
    @pytest.fixture
    def ollama_chat_response(self):
        """Sample Ollama chat API response."""
        return {
            "model": "qwen2.5:3b",
            "message": {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!"
            },
            "done": True,
            "prompt_eval_count": 25,
            "eval_count": 12,
        }
    
    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.model == "qwen2.5:3b"
        assert provider.api_url == "http://localhost:11434"
        assert provider.timeout == 120.0
    
    def test_initialization_with_custom_url(self):
        """Test initialization with custom URL."""
        provider = OllamaProvider(
            model="llama3.1:8b",
            api_url="http://custom-server:8080/",
            timeout=60.0,
        )
        assert provider.api_url == "http://custom-server:8080"  # Trailing slash removed
        assert provider.timeout == 60.0
    
    @pytest.mark.asyncio
    async def test_generate_success(self, provider, sample_messages, ollama_chat_response):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ollama_chat_response
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(provider._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await provider.generate(sample_messages)
            
            assert isinstance(result, LLMResponse)
            assert result.content == "I'm doing well, thank you for asking!"
            assert result.model == "qwen2.5:3b"
            assert result.finish_reason == "stop"
            assert result.usage["prompt_tokens"] == 25
            assert result.usage["completion_tokens"] == 12
            assert result.usage["total_tokens"] == 37
            assert result.latency_ms is not None  # Latency is measured even in mock
            
            # Verify request payload
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args.kwargs["json"]["model"] == "qwen2.5:3b"
            assert call_args.kwargs["json"]["stream"] == False
    
    @pytest.mark.asyncio
    async def test_generate_with_config(self, provider, sample_messages, ollama_chat_response):
        """Test generation with custom config."""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            stop_sequences=["END", "STOP"],
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ollama_chat_response
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(provider._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            await provider.generate(sample_messages, config)
            
            # Verify config was passed correctly
            call_args = mock_post.call_args
            options = call_args.kwargs["json"]["options"]
            assert options["num_predict"] == 512
            assert options["temperature"] == 0.5
            assert options["top_p"] == 0.95
            assert options["top_k"] == 40
            assert options["repeat_penalty"] == 1.2
            assert options["stop"] == ["END", "STOP"]
    
    @pytest.mark.asyncio
    async def test_generate_http_error(self, provider, sample_messages):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = '{"error":"model not found"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )
        
        with patch.object(provider._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            with pytest.raises(httpx.HTTPStatusError):
                await provider.generate(sample_messages)
    
    @pytest.mark.asyncio
    async def test_generate_connection_error(self, provider, sample_messages):
        """Test handling of connection errors."""
        with patch.object(provider._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(httpx.RequestError):
                await provider.generate(sample_messages)
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider):
        """Test health check when server is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch.object(provider._client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await provider.health_check()
            assert result == True
            mock_get.assert_called_with("http://localhost:11434/api/tags")
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, provider):
        """Test health check when server is not responding."""
        with patch.object(provider._client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            
            result = await provider.health_check()
            assert result == False
    
    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:3b"},
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"},
            ]
        }
        
        with patch.object(provider._client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            models = await provider.list_models()
            assert models == ["qwen2.5:3b", "llama3.1:8b", "mistral:7b"]
    
    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test list_models when server errors."""
        with patch.object(provider._client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Server error")
            
            models = await provider.list_models()
            assert models == []
    
    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test closing the client."""
        with patch.object(provider._client, 'aclose', new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()


class TestBaseMessageHelpers:
    """Test message helper functions."""
    
    def test_system_message(self):
        """Test system message creation."""
        msg = system_message("You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."
        assert msg.to_dict() == {"role": "system", "content": "You are helpful."}
    
    def test_user_message(self):
        """Test user message creation."""
        msg = user_message("Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
    
    def test_assistant_message(self):
        """Test assistant message creation."""
        msg = assistant_message("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"


class TestGenerationConfig:
    """Test GenerationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.stop_sequences == []
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=2048,
            temperature=0.3,
            stop_sequences=["END"],
        )
        assert config.max_tokens == 2048
        assert config.temperature == 0.3
        assert config.stop_sequences == ["END"]
    
    def test_to_dict(self):
        """Test config serialization."""
        config = GenerationConfig(max_tokens=512, stop_sequences=["STOP"])
        d = config.to_dict()
        assert d["max_tokens"] == 512
        assert d["stop"] == ["STOP"]


class TestLLMResponse:
    """Test LLMResponse."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Hello!",
            model="qwen2.5:3b",
        )
        assert response.content == "Hello!"
        assert response.model == "qwen2.5:3b"
        assert response.finish_reason is None
        assert response.usage is None
        assert response.latency_ms is None
    
    def test_full_response(self):
        """Test response with all fields."""
        response = LLMResponse(
            content="Hello!",
            model="qwen2.5:3b",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=250.5,
        )
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15
        assert response.latency_ms == 250.5
    
    def test_token_properties(self):
        """Test token count properties."""
        response = LLMResponse(
            content="Hello!",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.tokens_used == 15
    
    def test_token_properties_no_usage(self):
        """Test token properties when usage is None."""
        response = LLMResponse(content="Hello!", model="test")
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.tokens_used == 0


class TestLLMProviderFactory:
    """Test LLMProviderFactory."""
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        with patch("src.providers.llm.factory.load_model_config") as mock_config:
            mock_config.return_value = {}
            
            provider = LLMProviderFactory.create(
                provider_type="ollama",
                model="qwen2.5:3b",
            )
            
            assert isinstance(provider, OllamaProvider)
            assert provider.model == "qwen2.5:3b"
    
    def test_create_from_config(self):
        """Test creating provider from config file."""
        with patch("src.providers.llm.factory.load_model_config") as mock_config, \
             patch("src.providers.llm.factory.get_settings") as mock_settings:
            
            mock_config.return_value = {
                "providers": {
                    "llm": {
                        "provider": "ollama",
                        "model": "llama3.1:8b",
                    }
                }
            }
            mock_settings.return_value = MagicMock(
                ollama_api_url="http://localhost:11434",
            )
            
            provider = LLMProviderFactory.create()
            
            assert isinstance(provider, OllamaProvider)
            assert provider.model == "llama3.1:8b"
    
    def test_create_invalid_provider(self):
        """Test creating invalid provider raises error."""
        with patch("src.providers.llm.factory.load_model_config") as mock_config:
            mock_config.return_value = {}
            
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                LLMProviderFactory.create(provider_type="invalid_provider")
    
    @pytest.mark.asyncio
    async def test_create_with_fallback_primary_healthy(self):
        """Test fallback when primary is healthy."""
        with patch("src.providers.llm.factory.load_model_config") as mock_config, \
             patch("src.providers.llm.factory.get_settings") as mock_settings:
            
            mock_config.return_value = {
                "providers": {"llm": {"provider": "ollama", "model": "qwen2.5:3b"}}
            }
            mock_settings.return_value = MagicMock(
                ollama_api_url="http://localhost:11434",
            )
            
            with patch.object(OllamaProvider, 'health_check', new_callable=AsyncMock) as mock_health:
                mock_health.return_value = True
                
                provider = await LLMProviderFactory.create_with_fallback()
                
                assert isinstance(provider, OllamaProvider)
                mock_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_with_fallback_primary_unhealthy(self):
        """Test fallback when primary is unhealthy."""
        # Reset global provider
        import src.providers.llm.factory as factory_module
        factory_module._llm_provider = None
        
        with patch("src.providers.llm.factory.load_model_config") as mock_config, \
             patch("src.providers.llm.factory.get_settings") as mock_settings:
            
            mock_config.return_value = {
                "providers": {"llm": {"provider": "vllm", "model": "meta-llama/Llama-3.1-8B"}}
            }
            mock_settings.return_value = MagicMock(
                vllm_api_url="http://localhost:8001/v1",
                vllm_api_key=None,
                ollama_api_url="http://localhost:11434",
            )
            
            # VLLMProvider will fail health check, OllamaProvider will succeed
            call_count = [0]
            async def mock_health(self):
                call_count[0] += 1
                # First call is vLLM (fails), second is Ollama (succeeds)
                if isinstance(self, OllamaProvider):
                    return True
                return False
            
            with patch("src.providers.llm.vllm_provider.VLLMProvider.health_check", mock_health), \
                 patch.object(OllamaProvider, 'health_check', new_callable=AsyncMock) as ollama_health:
                ollama_health.return_value = True
                
                provider = await LLMProviderFactory.create_with_fallback()
                
                # Should fall back to Ollama
                assert isinstance(provider, OllamaProvider)


class TestLLMProviderSingleton:
    """Test global provider singleton behavior."""
    
    def test_get_llm_provider_sync(self):
        """Test synchronous provider getter."""
        # Reset global provider
        import src.providers.llm.factory as factory_module
        factory_module._llm_provider = None
        
        with patch("src.providers.llm.factory.load_model_config") as mock_config, \
             patch("src.providers.llm.factory.get_settings") as mock_settings:
            
            mock_config.return_value = {
                "providers": {"llm": {"provider": "ollama", "model": "qwen2.5:3b"}}
            }
            mock_settings.return_value = MagicMock(
                ollama_api_url="http://localhost:11434",
            )
            
            provider1 = get_llm_provider_sync()
            provider2 = get_llm_provider_sync()
            
            # Should return same instance
            assert provider1 is provider2
            assert isinstance(provider1, OllamaProvider)
