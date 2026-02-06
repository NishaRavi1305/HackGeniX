"""
Tests for Voice providers (STT and TTS).

Tests the Whisper STT and pyttsx3 TTS implementations.
"""
import pytest
import numpy as np
import wave
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.providers.stt.whisper_provider import (
    WhisperSTTProvider,
    WhisperModel,
    TranscriptionResult,
    TranscriptionSegment,
    get_whisper_provider,
)
from src.providers.tts.pyttsx3_provider import (
    Pyttsx3TTSProvider,
    VoiceGender,
    VoiceInfo,
    SynthesisResult,
    get_tts_provider,
)


# ============== Test Fixtures ==============

@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model."""
    model = Mock()
    model.transcribe.return_value = {
        "text": "Hello, this is a test transcription.",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Hello, this is a test transcription.",
                "avg_logprob": -0.3,
            }
        ],
    }
    return model


@pytest.fixture
def sample_audio_array():
    """Create a sample audio array (1 second of silence at 16kHz)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sample_wav_bytes():
    """Create sample WAV file bytes."""
    # Create a simple WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(16000)
        # Write 1 second of silence
        wav.writeframes(b'\x00' * 32000)
    
    buffer.seek(0)
    return buffer.read()


# ============== WhisperSTTProvider Tests ==============

class TestWhisperSTTProvider:
    """Tests for the Whisper STT provider."""
    
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    def test_initialization_cuda(self, mock_torch, mock_whisper):
        """Test provider initialization with CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_whisper.load_model.return_value = Mock()
        
        provider = WhisperSTTProvider(model_name="base")
        
        assert provider.model_name == "base"
        assert provider.device == "cuda"
        mock_whisper.load_model.assert_called_once_with("base", device="cuda")
    
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    def test_initialization_cpu_fallback(self, mock_torch, mock_whisper):
        """Test provider falls back to CPU when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = Mock()
        
        provider = WhisperSTTProvider(model_name="base")
        
        assert provider.device == "cpu"
        mock_whisper.load_model.assert_called_once_with("base", device="cpu")
    
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    def test_initialization_explicit_device(self, mock_torch, mock_whisper):
        """Test provider with explicit device specification."""
        mock_whisper.load_model.return_value = Mock()
        
        provider = WhisperSTTProvider(model_name="small", device="cpu")
        
        assert provider.device == "cpu"
        mock_whisper.load_model.assert_called_once_with("small", device="cpu")
    
    @pytest.mark.asyncio
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    async def test_transcribe_numpy_array(self, mock_torch, mock_whisper, sample_audio_array, mock_whisper_model):
        """Test transcription from numpy array."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = mock_whisper_model
        
        provider = WhisperSTTProvider(model_name="base")
        result = await provider.transcribe(sample_audio_array)
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."
        assert result.language == "en"
        assert len(result.segments) == 1
    
    @pytest.mark.asyncio
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    @patch('src.providers.stt.whisper_provider.sf')
    async def test_transcribe_file_path(self, mock_sf, mock_torch, mock_whisper, mock_whisper_model):
        """Test transcription from file path."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = mock_whisper_model
        mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        
        provider = WhisperSTTProvider(model_name="base")
        result = await provider.transcribe("/path/to/audio.wav")
        
        assert isinstance(result, TranscriptionResult)
        mock_sf.read.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    async def test_transcribe_with_prompt(self, mock_torch, mock_whisper, sample_audio_array, mock_whisper_model):
        """Test transcription with initial prompt."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = mock_whisper_model
        
        provider = WhisperSTTProvider(model_name="base")
        result = await provider.transcribe(
            sample_audio_array,
            prompt="Technical interview about Python and APIs"
        )
        
        # Verify prompt was passed
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("initial_prompt") == "Technical interview about Python and APIs"
    
    @pytest.mark.asyncio
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    async def test_transcribe_with_interview_context(self, mock_torch, mock_whisper, sample_audio_array, mock_whisper_model):
        """Test transcription with interview context."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = mock_whisper_model
        
        provider = WhisperSTTProvider(model_name="base")
        result = await provider.transcribe_with_interview_context(
            sample_audio_array,
            context="Backend engineer interview",
            technical_terms=["Python", "FastAPI", "PostgreSQL"],
        )
        
        assert isinstance(result, TranscriptionResult)
        # Verify prompt contains context
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        prompt = call_kwargs.get("initial_prompt", "")
        assert "technical interview" in prompt.lower()
    
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    def test_get_model_info(self, mock_torch, mock_whisper):
        """Test getting model information."""
        mock_torch.cuda.is_available.return_value = True
        mock_whisper.load_model.return_value = Mock()
        
        provider = WhisperSTTProvider(model_name="small", language="en")
        info = provider.get_model_info()
        
        assert info["model_name"] == "small"
        assert info["device"] == "cuda"
        assert info["language"] == "en"
    
    def test_resample(self):
        """Test audio resampling."""
        # Create a simple test without loading full model
        with patch('src.providers.stt.whisper_provider.whisper'):
            with patch('src.providers.stt.whisper_provider.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                
                provider = WhisperSTTProvider.__new__(WhisperSTTProvider)
                provider.model_name = "base"
                provider.device = "cpu"
                
                # Test resampling from 44100 to 16000
                audio_44k = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32)
                resampled = provider._resample(audio_44k, 44100, 16000)
                
                assert len(resampled) == 16000
                assert resampled.dtype == np.float32


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = TranscriptionResult(
            text="Test transcription",
            language="en",
            confidence=0.95,
            duration_seconds=5.0,
            segments=[{"id": 0, "text": "Test"}],
        )
        
        d = result.to_dict()
        
        assert d["text"] == "Test transcription"
        assert d["language"] == "en"
        assert d["confidence"] == 0.95
        assert d["duration_seconds"] == 5.0


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""
    
    def test_to_dict(self):
        """Test converting segment to dictionary."""
        segment = TranscriptionSegment(
            id=0,
            start=0.0,
            end=2.5,
            text="Hello world",
            confidence=0.9,
        )
        
        d = segment.to_dict()
        
        assert d["id"] == 0
        assert d["start"] == 0.0
        assert d["end"] == 2.5
        assert d["text"] == "Hello world"


# ============== Pyttsx3TTSProvider Tests ==============

class TestPyttsx3TTSProvider:
    """Tests for the pyttsx3 TTS provider."""
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_initialization(self, mock_pyttsx3):
        """Test TTS provider initialization."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider(rate=150, volume=0.8)
        
        assert provider.rate == 150
        assert provider.volume == 0.8
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_get_voices(self, mock_pyttsx3):
        """Test getting available voices."""
        mock_voice = Mock()
        mock_voice.id = "voice1"
        mock_voice.name = "Microsoft David"
        mock_voice.languages = ["en_US"]
        
        mock_engine = Mock()
        mock_engine.getProperty.return_value = [mock_voice]
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider()
        voices = provider.get_voices()
        
        assert len(voices) == 1
        assert voices[0].id == "voice1"
        assert voices[0].name == "Microsoft David"
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_set_voice(self, mock_pyttsx3):
        """Test setting voice."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider()
        provider.set_voice("new_voice_id")
        
        assert provider.voice_id == "new_voice_id"
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_set_rate(self, mock_pyttsx3):
        """Test setting speech rate."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider(rate=150)
        provider.set_rate(180)
        
        assert provider.rate == 180
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_set_volume(self, mock_pyttsx3):
        """Test setting volume with clamping."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider()
        
        # Test normal value
        provider.set_volume(0.7)
        assert provider.volume == 0.7
        
        # Test clamping above 1.0
        provider.set_volume(1.5)
        assert provider.volume == 1.0
        
        # Test clamping below 0.0
        provider.set_volume(-0.5)
        assert provider.volume == 0.0
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_get_provider_info(self, mock_pyttsx3):
        """Test getting provider info."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider(rate=160, volume=0.9)
        info = provider.get_provider_info()
        
        assert info["provider"] == "pyttsx3"
        assert info["rate"] == 160
        assert info["volume"] == 0.9
    
    @pytest.mark.asyncio
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    async def test_synthesize_empty_text_raises(self, mock_pyttsx3):
        """Test that empty text raises error."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        provider = Pyttsx3TTSProvider()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await provider.synthesize("")


class TestVoiceInfo:
    """Tests for VoiceInfo dataclass."""
    
    def test_to_dict(self):
        """Test converting voice info to dictionary."""
        voice = VoiceInfo(
            id="voice1",
            name="Test Voice",
            languages=["en_US", "en_GB"],
            gender="female",
            age="adult",
        )
        
        d = voice.to_dict()
        
        assert d["id"] == "voice1"
        assert d["name"] == "Test Voice"
        assert d["languages"] == ["en_US", "en_GB"]
        assert d["gender"] == "female"


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""
    
    def test_to_dict(self):
        """Test converting synthesis result to dictionary."""
        result = SynthesisResult(
            audio_data=b"fake_audio_data",
            sample_rate=22050,
            duration_seconds=3.5,
            text="Hello world",
            voice_id="voice1",
        )
        
        d = result.to_dict()
        
        assert d["sample_rate"] == 22050
        assert d["duration_seconds"] == 3.5
        assert d["text"] == "Hello world"
        assert d["audio_size_bytes"] == len(b"fake_audio_data")


# ============== Singleton Tests ==============

class TestSingletons:
    """Tests for singleton pattern implementations."""
    
    @patch('src.providers.stt.whisper_provider.whisper')
    @patch('src.providers.stt.whisper_provider.torch')
    def test_get_whisper_provider_singleton(self, mock_torch, mock_whisper):
        """Test that get_whisper_provider returns same instance."""
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = Mock()
        
        # Reset singleton
        import src.providers.stt.whisper_provider as stt_module
        stt_module._whisper_provider = None
        
        provider1 = get_whisper_provider()
        provider2 = get_whisper_provider()
        
        assert provider1 is provider2
    
    @patch('src.providers.tts.pyttsx3_provider.pyttsx3')
    def test_get_tts_provider_singleton(self, mock_pyttsx3):
        """Test that get_tts_provider returns same instance."""
        mock_engine = Mock()
        mock_engine.getProperty.return_value = []
        mock_pyttsx3.init.return_value = mock_engine
        
        # Reset singleton
        import src.providers.tts.pyttsx3_provider as tts_module
        tts_module._tts_provider = None
        
        provider1 = get_tts_provider()
        provider2 = get_tts_provider()
        
        assert provider1 is provider2


# ============== Enum Tests ==============

class TestEnums:
    """Tests for voice-related enums."""
    
    def test_whisper_model_enum(self):
        """Test WhisperModel enum values."""
        assert WhisperModel.TINY == "tiny"
        assert WhisperModel.BASE == "base"
        assert WhisperModel.SMALL == "small"
        assert WhisperModel.MEDIUM == "medium"
        assert WhisperModel.LARGE == "large"
    
    def test_voice_gender_enum(self):
        """Test VoiceGender enum values."""
        assert VoiceGender.MALE == "male"
        assert VoiceGender.FEMALE == "female"
        assert VoiceGender.NEUTRAL == "neutral"
