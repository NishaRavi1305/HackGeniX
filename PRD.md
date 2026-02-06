# Product Requirements Document
## AI Agentic Interviewer System

**Project Owner:** Hackgenix Tech Private Limited  
**Document Version:** 2.0  
**Last Updated:** February 2026  
**Project Type:** Autonomous Interview Agent with Assessment & Reporting

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technical Stack](#3-technical-stack)
4. [Agent Design](#4-agent-design)
5. [Assessment Engine](#5-assessment-engine)
6. [Report Generation](#6-report-generation)
7. [API Specifications](#7-api-specifications)
8. [Data Architecture](#8-data-architecture)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Security & Compliance](#10-security--compliance)
11. [Deployment Strategy](#11-deployment-strategy)
12. [Timeline & Milestones](#12-timeline--milestones)

---

## 1. Executive Summary

### 1.1 Project Objective
Build an autonomous AI interviewer that conducts full technical + behavioral interviews by analyzing job descriptions and candidate resumes, then generates comprehensive hiring reports with scoring and recommendations.

### 1.2 Comparison: BerriBot Model
**BerriBot's approach** (per their documentation):
- Structured interview flow based on job requirements
- Real-time candidate assessment scoring
- Automated report generation
- Integration via API for external platforms

**Our differentiation:**
- Multi-stage interview agent (screening → technical → behavioral → wrap-up)
- Resume-JD semantic matching for personalized question generation
- Hallucination-proof evaluation (grounded in rubrics)
- Export-ready hiring reports (PDF with scoring breakdown)

### 1.3 Scope Boundaries
**IN SCOPE (Backend Responsibility):**
- Backend interview orchestration logic
- Question generation from JD + resume parsing
- Real-time scoring engine (technical + behavioral)
- Report generation API (JSON + PDF)
- Model evaluation metrics
- **STT (Speech-to-Text) processing** - convert candidate audio → text
- **TTS (Text-to-Speech) synthesis** - convert agent questions → audio for Unity avatar

**OUT OF SCOPE:**
- Unity frontend/avatar implementation (client responsibility)
- Audio capture/playback (Unity's AudioSource handles this)
- Candidate authentication/scheduling system
- ATS integration

**CLARIFICATION ON VOICE PIPELINE:**
Unity captures microphone input → sends audio bytes to backend → STT converts to text → Agent processes → TTS generates response audio → Unity plays through avatar

### 1.4 Success Metrics
- **Interview Completion Rate:** >85% (candidates finish all stages)
- **Question Relevance:** >90% aligned with JD requirements (human eval)
- **Scoring Consistency:** Inter-rater reliability >0.75 vs human interviewers
- **Report Generation Time:** <30s post-interview
- **API Uptime:** 99.5% during demo period

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         UNITY CLIENT                              │
│  • Avatar rendering & animation                                   │
│  • Microphone capture (AudioClip → byte[] WAV)                   │
│  • Audio playback (AudioSource for TTS responses)                │
│  • UI for interview progress                                      │
└────────────────┬─────────────────────────────────────────────────┘
                 │ REST API (multipart audio) / WebSocket (streaming)
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     BACKEND ORCHESTRATOR                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Voice Processing Layer (NEW)                   │ │
│  │                                                              │ │
│  │  ┌──────────────────┐         ┌──────────────────────┐    │ │
│  │  │ STT Engine       │         │ TTS Engine           │    │ │
│  │  │ (Whisper-large)  │         │ (XTTS-v2 / Piper)    │    │ │
│  │  │                  │         │                      │    │ │
│  │  │ Audio → Text     │         │ Text → Audio         │    │ │
│  │  │ + Diarization    │         │ Voice cloning        │    │ │
│  │  │ + Timestamps     │         │ Multi-language       │    │ │
│  │  └──────────────────┘         └──────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────┘ │
│          │                                    │                   │
│  ┌───────▼────────────────────────────────────▼───────────────┐ │
│  │              Interview Agent Controller                     │ │
│  │  • State machine (screening → tech → behavioral → done)    │ │
│  │  • Context window management                                │ │
│  │  • Dynamic question generation                              │ │
│  └───────┬────────────────────────────────────────────────────┘ │
│          │                                                         │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │           Document Processing Pipeline                      │ │
│  │                                                              │ │
│  │  ┌──────────────┐      ┌────────────────────────────┐     │ │
│  │  │ Resume       │      │ Job Description            │     │ │
│  │  │ Parser       │──────│ Extractor                  │     │ │
│  │  │              │      │                            │     │ │
│  │  │ • Skills     │      │ • Required skills          │     │ │
│  │  │ • Experience │      │ • Experience level         │     │ │
│  │  │ • Education  │      │ • Responsibilities         │     │ │
│  │  └──────────────┘      └────────────────────────────┘     │ │
│  │          │                            │                     │ │
│  │          └────────────┬───────────────┘                     │ │
│  │                       ▼                                      │ │
│  │          ┌────────────────────────────┐                     │ │
│  │          │   Semantic Matcher         │                     │ │
│  │          │   (Embeddings + Scoring)   │                     │ │
│  │          └────────────────────────────┘                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│          │                                                         │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │              Question Generation Engine                     │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Technical Questions (LLM-generated)                  │ │ │
│  │  │  • Coding problems (based on JD tech stack)          │ │ │
│  │  │  • System design (seniority-appropriate)             │ │ │
│  │  │  • Domain-specific scenarios                         │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Behavioral Questions (Template + LLM)               │ │ │
│  │  │  • STAR method probing                               │ │ │
│  │  │  • Culture fit assessment                            │ │ │
│  │  │  • Situational judgment                              │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│          │                                                         │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │                LLM Inference Layer                          │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Llama-3.1-8B-Instruct (4-bit quantized)             │ │ │
│  │  │  • Interview conversation                             │ │ │
│  │  │  • Follow-up question generation                      │ │ │
│  │  │  • Clarification requests                             │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│          │                                                         │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │              Real-Time Assessment Engine                    │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Answer Evaluator (LLM-as-Judge)                     │ │ │
│  │  │  • Rubric-based scoring (1-5 scale)                  │ │ │
│  │  │  • Key point extraction                               │ │ │
│  │  │  • Red flag detection                                 │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Behavioral Analyzer                                  │ │ │
│  │  │  • STAR completeness check                           │ │ │
│  │  │  • Sentiment analysis                                 │ │ │
│  │  │  • Communication quality metrics                      │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│          │                                                         │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │              Report Generation Engine                       │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Score Aggregator                                     │ │ │
│  │  │  • Technical: weighted avg by question difficulty    │ │ │
│  │  │  • Behavioral: STAR completeness + culture fit       │ │ │
│  │  │  • Overall: composite score with hiring rec          │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  PDF Generator (ReportLab)                           │ │ │
│  │  │  • Executive summary                                  │ │ │
│  │  │  • Question-by-question breakdown                     │ │ │
│  │  │  • Strengths/weaknesses                              │ │ │
│  │  │  • Hiring recommendation                             │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
┌──────────────────┐             ┌──────────────────┐
│  MONGODB ATLAS   │             │   S3 STORAGE     │
│                  │             │                  │
│  • interviews    │             │  • Resume PDFs   │
│  • assessments   │             │  • Report PDFs   │
│  • candidates    │             │  • Audio logs    │
│  • questions     │             │    (optional)    │
└──────────────────┘             └──────────────────┘
```

### 2.2 Interview Agent State Machine

```
START
  │
  ├──> INITIALIZATION
  │      • Parse resume + JD
  │      • Generate interview plan
  │      • Calculate gap analysis
  │      └──> Duration: 30s
  │
  ├──> SCREENING (5-7 min)
  │      • Verify basic qualifications
  │      • Communication assessment
  │      • Role expectations alignment
  │      └──> PASS → Continue | FAIL → Early termination
  │
  ├──> TECHNICAL DEEP DIVE (15-25 min)
  │      • Coding/problem-solving (if applicable)
  │      • System design (senior roles)
  │      • Domain knowledge (role-specific)
  │      • Adaptive difficulty (based on performance)
  │      └──> Real-time scoring per answer
  │
  ├──> BEHAVIORAL ASSESSMENT (10-15 min)
  │      • STAR-based scenario questions
  │      • Culture fit probes
  │      • Team dynamics scenarios
  │      • Conflict resolution
  │      └──> Sentiment + completeness scoring
  │
  ├──> WRAP-UP (2-3 min)
  │      • Candidate questions
  │      • Next steps communication
  │      • Feedback collection
  │
  └──> REPORT GENERATION
         • Aggregate scores
         • Generate insights
         • PDF compilation
         └──> Duration: 20-30s
```

### 2.3 Data Flow

```
Resume PDF + JD JSON
    │
    ├─→ Document Processing
    │     • Text extraction (pypdf2)
    │     • Entity recognition (spaCy)
    │     • Skill embedding (bge-large-en-v1.5)
    │
    ├─→ Semantic Matching
    │     • JD requirements vs resume skills
    │     • Experience level alignment
    │     • Gap identification
    │     └─→ Match Score (0-100)
    │
    ├─→ Interview Plan Generation (LLM)
    │     Prompt: "Given JD requiring {skills} and candidate with {resume_skills},
    │              generate 8 technical + 5 behavioral questions. Focus gaps: {gaps}"
    │     └─→ Structured Question List with Rubrics
    │
    ├─→ Interview Execution Loop
    │     FOR each question:
    │       • Agent asks question
    │       • Candidate responds (via Unity → API)
    │       • LLM evaluates answer against rubric
    │       • Score recorded (1-5 scale)
    │       • Follow-up generated if score < 3
    │
    ├─→ Real-Time Score Aggregation
    │     • Technical Score = Σ(question_score × difficulty_weight) / total_weight
    │     • Behavioral Score = Σ(STAR_completeness × relevance) / num_questions
    │     • Communication Score = avg(clarity, conciseness, professionalism)
    │
    └─→ Report Generation
          • JSON structure (for Unity display)
          • PDF export (hiring team download)
          • MongoDB persistence
```

---

## 3. Technical Stack

### 3.1 Core Backend

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Web Framework** | FastAPI | 0.104.1 | Async, WebSocket support for real-time interviews |
| **LLM** | Llama-3.1-8B-Instruct | 8B (4-bit) | Instruction following, reasoning capability (Meta, 2024). **Swappable** - see Section 3.4. Recommended upgrade: Qwen2.5-32B-Instruct (Apache 2.0) for higher quality. |
| **Inference** | vLLM | 0.2.7 | Low latency for conversational flow |
| **STT Model** | Whisper-large-v3 | 1550M params | SOTA accuracy (OpenAI, 2024), multilingual, 8% WER on clean speech |
| **TTS Model** | XTTS-v2 | 400M params | Voice cloning, emotional control (Coqui AI, 2023), 24kHz output. **License:** Coqui Public Model License (CPML) - review at coqui.ai/cpml for commercial use. **Swappable** alternatives: Piper TTS (MIT), StyleTTS2 (MIT). |
| **Resume Parsing** | spaCy | 3.7.2 | NER for skills/experience extraction |
| **Embeddings** | BAAI/bge-large-en-v1.5 | 1024-dim | Semantic matching resume↔JD; MIT license; superior retrieval quality vs MiniLM |
| **PDF Generation** | ReportLab | 4.0.7 | Programmatic PDF creation |
| **Document DB** | MongoDB Atlas | 7.0 | Flexible schema for interview data |
| **Object Storage** | AWS S3 / MinIO | - | Resume + report + audio storage |

**Key Dependencies:**
```
Python 3.11
├── fastapi[all]==0.104.1
├── vllm==0.2.7
├── openai-whisper==20231117  # Or faster-whisper==0.10.0 (CTranslate2 backend)
├── TTS==0.21.1  # Coqui TTS for XTTS-v2
├── pydub==0.25.1  # Audio format conversion
├── soundfile==0.12.1  # Audio I/O
├── spacy==3.7.2
├── sentence-transformers==2.2.2
├── reportlab==4.0.7
├── motor==3.3.2 (async MongoDB)
├── pypdf2==3.0.1
├── python-multipart==0.0.6 (file uploads)
└── pydantic==2.5.0
```

### 3.2 Document Processing Stack

```python
# Resume parsing
spacy.load("en_core_web_trf")  # Transformer-based NER
# Extracts: PERSON, ORG, DATE, GPE, SKILL (custom training)

# PDF text extraction
pypdf2==3.0.1  # Fallback: pdfplumber for complex layouts

# Skill taxonomy
skills_taxonomy.json  # Pre-built: 5000+ tech skills with categories
# Source: ESCO (European Skills/Competences) + Stack Overflow taxonomy
```

### 3.3 Infrastructure

| Service | Tier | Cost/Month | Notes |
|---------|------|------------|-------|
| **GPU Compute** | Runpod RTX A4000 (spot) | ₹8,160 | For LLM + Whisper inference |
| **GPU Compute (TTS)** | Runpod RTX 3090 (spot) | ₹5,440 | Dedicated for XTTS real-time (optional, can share with LLM) |
| **MongoDB** | M10 (10GB) | ₹5,700 | |
| **S3 Storage** | 50GB | ₹750 | Resumes + reports + audio logs |
| **Redis** | 250MB (free tier) | ₹0 | Session management |
| **TOTAL (Single GPU)** | | **₹14,610** | If LLM + STT share GPU |
| **TOTAL (Dual GPU)** | | **₹20,050** | If TTS needs dedicated GPU for low latency |

**GPU Memory Requirements (Default Stack):**
- Llama-3.1-8B (4-bit): ~5GB VRAM
- Whisper-large-v3 (FP16): ~3GB VRAM
- XTTS-v2 (FP16): ~4GB VRAM
- bge-large-en-v1.5 (FP16): ~1.3GB VRAM
- **Total if co-hosted:** ~13.3GB (fits on A4000 16GB)
- **Recommendation:** Start single GPU, scale to dual if TTS latency >2s

**GPU Memory Requirements (Upgraded Stack - Qwen2.5-32B):**
- Qwen2.5-32B-Instruct (4-bit AWQ): ~18-20GB VRAM
- Whisper-large-v3 (FP16): ~3GB VRAM
- XTTS-v2 (FP16): ~4GB VRAM
- bge-large-en-v1.5 (FP16): ~1.3GB VRAM
- **Total if co-hosted:** ~26-28GB (requires RTX A6000 48GB or dual GPU setup)
- **Recommendation:** Use 48GB GPU or split LLM onto dedicated GPU

### 3.4 Modular Provider Architecture

The system is designed with a **plug-and-play architecture** allowing each AI component (LLM, Embeddings, STT, TTS) to be swapped independently without code changes. This enables the deploying organization to upgrade or downgrade models based on hardware availability, cost constraints, or quality requirements.

#### 3.4.1 Design Philosophy

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│                  (FastAPI + Orchestrator)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ Abstraction Layer (Provider Interfaces)
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
   │   LLM   │  │Embeddings│  │   STT   │  │   TTS    │
   │ Provider│  │ Provider │  │ Provider│  │ Provider │
   └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘
        │            │             │            │
   ┌────▼────┐  ┌────▼─────┐  ┌────▼────┐  ┌────▼─────┐
   │Qwen 32B │  │bge-large │  │ Whisper │  │ XTTS-v2  │
   │Llama 8B │  │MiniLM    │  │ faster- │  │ Piper    │
   │Mistral  │  │bge-small │  │ whisper │  │ StyleTTS │
   │  etc.   │  │  etc.    │  │  etc.   │  │  etc.    │
   └─────────┘  └──────────┘  └─────────┘  └──────────┘
```

**Key Principles:**
- **Configuration-driven:** Model selection via YAML config, not code changes
- **Interface contracts:** Each provider type implements a standard interface
- **Hot-swappable:** Models can be changed without restarting the entire system (with graceful reload)
- **Fallback support:** Define backup models for resilience

#### 3.4.2 Provider Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator

class LLMProvider(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: List[str] = None
    ) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text completion tokens"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, params, context length)"""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Return embedding vector dimensions"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        pass


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text providers"""
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Returns:
            {
                "text": "transcribed text",
                "segments": [...],  # Optional: word/segment timestamps
                "language": "en",
                "duration": 45.2,
                "confidence": 0.95  # Optional
            }
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return supported audio formats (wav, mp3, etc.)"""
        pass


class TTSProvider(ABC):
    """Abstract interface for Text-to-Speech providers"""
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        voice_id: str = "default",
        speed: float = 1.0
    ) -> bytes:
        """
        Synthesize speech from text
        
        Returns:
            WAV audio bytes
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks for real-time playback"""
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, str]]:
        """Return list of available voice options"""
        pass
```

#### 3.4.3 Configuration Schema

**Config file: `config/models.yaml`**

```yaml
# Model Configuration - Swap models by changing these values
# No code changes required

providers:
  llm:
    provider: "vllm"  # Options: vllm, ollama, llamacpp, openai-compatible
    model: "meta-llama/Llama-3.1-8B-Instruct"
    quantization: "awq-4bit"  # Options: none, gptq-4bit, awq-4bit, int8
    max_context_length: 8192
    gpu_memory_utilization: 0.85
    # Upgrade option (uncomment to use):
    # model: "Qwen/Qwen2.5-32B-Instruct-AWQ"
    # quantization: "awq-4bit"
    
  embeddings:
    provider: "sentence-transformers"  # Options: sentence-transformers, huggingface
    model: "BAAI/bge-large-en-v1.5"
    dimensions: 1024
    device: "cuda"
    # Downgrade option for lower VRAM:
    # model: "sentence-transformers/all-MiniLM-L6-v2"
    # dimensions: 384
    
  stt:
    provider: "faster-whisper"  # Options: whisper, faster-whisper
    model: "large-v3"
    compute_type: "int8"  # Options: float16, int8, int8_float16
    device: "cuda"
    language: "en"  # Force language or "auto" for detection
    vad_filter: true
    # Lighter option:
    # model: "medium"
    # compute_type: "int8"
    
  tts:
    provider: "coqui-xtts"  # Options: coqui-xtts, piper, styletts2
    model: "tts_models/multilingual/multi-dataset/xtts_v2"
    device: "cuda"
    reference_audio: "assets/interviewer_voice.wav"
    # Faster/lighter alternative:
    # provider: "piper"
    # model: "en_US-lessac-medium"
    
# Fallback configuration
fallbacks:
  tts:
    - provider: "piper"
      model: "en_US-lessac-medium"
      trigger: "latency > 3000ms"  # Use fallback if primary is slow

# Resource limits
resources:
  max_concurrent_stt: 4
  max_concurrent_tts: 2
  stt_timeout_ms: 10000
  tts_timeout_ms: 8000
```

**Environment variable overrides:**

```bash
# Override any config value via environment variables
export PROVIDER_LLM_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
export PROVIDER_EMBEDDINGS_MODEL="BAAI/bge-large-en-v1.5"
export PROVIDER_STT_MODEL="large-v3"
export PROVIDER_TTS_PROVIDER="piper"
```

#### 3.4.4 Provider Factory Pattern

```python
from config import load_model_config

class ProviderFactory:
    """Factory for creating provider instances based on configuration"""
    
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_llm_provider(cls) -> LLMProvider:
        if "llm" not in cls._instances:
            config = load_model_config()["providers"]["llm"]
            
            if config["provider"] == "vllm":
                from providers.llm.vllm_provider import VLLMProvider
                cls._instances["llm"] = VLLMProvider(
                    model=config["model"],
                    quantization=config.get("quantization"),
                    gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9)
                )
            elif config["provider"] == "ollama":
                from providers.llm.ollama_provider import OllamaProvider
                cls._instances["llm"] = OllamaProvider(model=config["model"])
            else:
                raise ValueError(f"Unknown LLM provider: {config['provider']}")
        
        return cls._instances["llm"]
    
    @classmethod
    def get_embedding_provider(cls) -> EmbeddingProvider:
        if "embeddings" not in cls._instances:
            config = load_model_config()["providers"]["embeddings"]
            
            from providers.embeddings.sentence_transformer_provider import SentenceTransformerProvider
            cls._instances["embeddings"] = SentenceTransformerProvider(
                model=config["model"],
                device=config.get("device", "cuda")
            )
        
        return cls._instances["embeddings"]
    
    @classmethod
    def get_stt_provider(cls) -> STTProvider:
        if "stt" not in cls._instances:
            config = load_model_config()["providers"]["stt"]
            
            if config["provider"] == "faster-whisper":
                from providers.stt.faster_whisper_provider import FasterWhisperProvider
                cls._instances["stt"] = FasterWhisperProvider(
                    model=config["model"],
                    compute_type=config.get("compute_type", "int8"),
                    device=config.get("device", "cuda")
                )
            elif config["provider"] == "whisper":
                from providers.stt.whisper_provider import WhisperProvider
                cls._instances["stt"] = WhisperProvider(model=config["model"])
        
        return cls._instances["stt"]
    
    @classmethod
    def get_tts_provider(cls) -> TTSProvider:
        if "tts" not in cls._instances:
            config = load_model_config()["providers"]["tts"]
            
            if config["provider"] == "coqui-xtts":
                from providers.tts.xtts_provider import XTTSProvider
                cls._instances["tts"] = XTTSProvider(
                    model=config["model"],
                    reference_audio=config.get("reference_audio")
                )
            elif config["provider"] == "piper":
                from providers.tts.piper_provider import PiperProvider
                cls._instances["tts"] = PiperProvider(model=config["model"])
        
        return cls._instances["tts"]
    
    @classmethod
    def reload_provider(cls, provider_type: str):
        """Hot-reload a specific provider (graceful restart)"""
        if provider_type in cls._instances:
            old_instance = cls._instances.pop(provider_type)
            # Cleanup old instance
            if hasattr(old_instance, 'cleanup'):
                old_instance.cleanup()
        # Next call to get_*_provider will create new instance
```

#### 3.4.5 Swap/Upgrade Matrix

| Component | Swap Action | Effort | Downtime | Notes |
|-----------|-------------|--------|----------|-------|
| **LLM** (8B → 32B) | Change `model` in config | Config only | ~30s reload | Ensure sufficient VRAM |
| **LLM** (32B → 8B) | Change `model` in config | Config only | ~30s reload | Frees VRAM |
| **Embeddings** | Change `model` + run re-index | Config + Script | 1-2 hours | **Requires re-embedding all documents** |
| **STT** (Whisper variant) | Change `model` in config | Config only | ~10s reload | No data migration |
| **TTS** (XTTS → Piper) | Change `provider` in config | Config only | ~10s reload | Different voice quality |

#### 3.4.6 Embeddings Re-indexing Utility

When changing the embeddings model, all stored vectors must be regenerated because:
1. Different models produce different vector dimensions (384 vs 1024)
2. Even same-dimension models have incompatible vector spaces

**Re-indexing script: `scripts/reindex_embeddings.py`**

```python
#!/usr/bin/env python3
"""
Re-index all embeddings when changing the embedding model.

Usage:
    python scripts/reindex_embeddings.py --model BAAI/bge-large-en-v1.5
    python scripts/reindex_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

async def reindex_embeddings(model_name: str, batch_size: int = 32):
    """Re-embed all documents with new model"""
    
    # Initialize new embedder
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    new_dimensions = embedder.get_sentence_embedding_dimension()
    print(f"New embedding dimensions: {new_dimensions}")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.interview_system
    
    # Collections to re-index
    collections_to_update = [
        ("interview_sessions", "resume.parsed_data.summary_embedding", "resume.parsed_data.raw_text"),
        ("interview_sessions", "job_description.jd_embedding", "job_description.description"),
    ]
    
    for collection_name, embedding_field, text_field in collections_to_update:
        collection = db[collection_name]
        
        # Count documents
        total = await collection.count_documents({})
        print(f"\nRe-indexing {collection_name}.{embedding_field} ({total} documents)")
        
        # Process in batches
        cursor = collection.find({}, {text_field: 1})
        batch = []
        doc_ids = []
        
        async for doc in tqdm(cursor, total=total):
            # Extract text to embed
            text = get_nested_value(doc, text_field)
            if not text:
                continue
            
            batch.append(text[:2000])  # Limit text length
            doc_ids.append(doc["_id"])
            
            if len(batch) >= batch_size:
                # Generate embeddings
                embeddings = embedder.encode(batch, show_progress_bar=False)
                
                # Update documents
                for doc_id, embedding in zip(doc_ids, embeddings):
                    await collection.update_one(
                        {"_id": doc_id},
                        {"$set": {embedding_field: embedding.tolist()}}
                    )
                
                batch = []
                doc_ids = []
        
        # Process remaining
        if batch:
            embeddings = embedder.encode(batch, show_progress_bar=False)
            for doc_id, embedding in zip(doc_ids, embeddings):
                await collection.update_one(
                    {"_id": doc_id},
                    {"$set": {embedding_field: embedding.tolist()}}
                )
    
    # Update vector index dimensions in MongoDB (if using Atlas Vector Search)
    print(f"\n[ACTION REQUIRED] Update MongoDB vector index to dimensions={new_dimensions}")
    print("Run the following in MongoDB shell:")
    print(f"""
db.runCommand({{
  "createSearchIndexes": "interview_sessions",
  "indexes": [{{
    "name": "embedding_index",
    "definition": {{
      "mappings": {{
        "dynamic": true,
        "fields": {{
          "resume.parsed_data.summary_embedding": {{
            "type": "knnVector",
            "dimensions": {new_dimensions},
            "similarity": "cosine"
          }}
        }}
      }}
    }}
  }}]
}})
""")
    
    print("\nRe-indexing complete!")

def get_nested_value(doc: dict, field_path: str):
    """Get nested field value using dot notation"""
    keys = field_path.split(".")
    value = doc
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="New embedding model name")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    asyncio.run(reindex_embeddings(args.model, args.batch_size))
```

#### 3.4.7 Hardware Requirements by Tier

| Tier | LLM | Embeddings | STT | TTS | Total VRAM | Recommended GPU |
|------|-----|------------|-----|-----|------------|-----------------|
| **Lite** | Llama-3.1-8B (4-bit) | bge-small-en-v1.5 | Whisper-medium | Piper | ~10GB | RTX 3080/4080 |
| **Standard** | Llama-3.1-8B (4-bit) | bge-large-en-v1.5 | Whisper-large-v3 | XTTS-v2 | ~13GB | RTX A4000 16GB |
| **Enhanced** | Qwen2.5-14B (4-bit) | bge-large-en-v1.5 | Whisper-large-v3 | XTTS-v2 | ~18GB | RTX 4090 24GB |
| **Premium** | Qwen2.5-32B (4-bit) | bge-large-en-v1.5 | Whisper-large-v3 | XTTS-v2 | ~28GB | RTX A6000 48GB |
| **Enterprise** | Qwen2.5-32B (4-bit) | bge-large-en-v1.5 | Whisper-large-v3 | XTTS-v2 | ~28GB | 2× RTX 4090 (tensor parallel) |

**Tier Selection Guide:**
- **Lite:** Development/testing, budget constraints, lower interview quality acceptable
- **Standard:** Production MVP, good balance of cost and quality
- **Enhanced:** Higher quality responses, better reasoning for senior roles
- **Premium:** Best single-GPU quality, complex technical interviews
- **Enterprise:** Maximum quality with redundancy, high-volume deployments

---

## 4. Voice Processing Pipeline

### 4.1 Speech-to-Text (STT) Engine

**Model Selection: Whisper-large-v3**

**Rationale:**
- **Accuracy:** 8-10% WER on clean speech, 15-20% on accented English (OpenAI benchmarks)
- **Multilingual:** Supports 99 languages (future-proof for global hiring)
- **Robustness:** Handles background noise, filler words, accents better than alternatives
- **Open Source:** MIT license, production-ready
- **Source:** OpenAI (2023), validated in production by Hugging Face, Deepgram alternatives

**Architecture:**
```python
import whisper
from faster_whisper import WhisperModel  # CTranslate2 backend (4x faster)

class STTEngine:
    def __init__(self):
        # Use faster-whisper for production (quantized INT8)
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="int8",  # 4x faster, <2% accuracy loss
            num_workers=2  # Parallel batch processing
        )
        
    async def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Convert audio to text with timestamps
        
        Returns:
        {
            "text": "full transcription",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello, my name is..."},
                ...
            ],
            "language": "en",
            "duration": 45.2
        }
        """
        # Save audio temporarily
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Transcribe with timestamps
        segments, info = self.model.transcribe(
            temp_path,
            beam_size=5,  # Trade-off: accuracy vs speed
            vad_filter=True,  # Voice activity detection (removes silence)
            language="en"  # Force English for consistency
        )
        
        # Aggregate segments
        full_text = ""
        segment_list = []
        for segment in segments:
            full_text += segment.text + " "
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "text": full_text.strip(),
            "segments": segment_list,
            "language": info.language,
            "duration": info.duration
        }
```

**Performance Characteristics:**
- **Latency:** ~1.5s for 10s audio clip (on RTX A4000)
- **Throughput:** Can process 4-5 audio clips simultaneously
- **Accuracy:** 92% on LibriSpeech test-clean, 85% on accented speech

**Error Handling:**
```python
async def transcribe_with_fallback(self, audio_bytes: bytes) -> dict:
    """Handle edge cases: too short, too noisy, silence"""
    
    # Check audio duration
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    duration_ms = len(audio)
    
    if duration_ms < 500:  # Less than 0.5s
        return {"text": "", "error": "audio_too_short"}
    
    if duration_ms > 180000:  # More than 3 minutes
        return {"text": "", "error": "audio_too_long"}
    
    # Check if mostly silence
    if audio.dBFS < -40:  # Very quiet
        return {"text": "", "error": "mostly_silence"}
    
    try:
        result = await self.transcribe(audio_bytes)
        
        # Post-process: remove filler words for cleaner analysis
        result['text_clean'] = self._remove_fillers(result['text'])
        
        return result
    except Exception as e:
        logger.error(f"STT failed: {str(e)}")
        return {"text": "", "error": "transcription_failed"}
```

### 4.2 Text-to-Speech (TTS) Engine

**Model Selection: XTTS-v2 (Coqui AI)**

**Rationale:**
- **Voice Cloning:** Can clone target voice from 6s sample (create consistent interviewer persona)
- **Emotional Control:** Can inject professional, friendly, or empathetic tones
- **Quality:** 24kHz output, natural prosody
- **Latency:** ~2s for 50-word sentence (acceptable for conversational flow)
- **License:** AGPL (acceptable for backend service, not distributed)
- **Source:** Coqui AI (2023), production-tested by HeyGen, ElevenLabs alternatives

**Alternative (Faster, Lower Quality):** Piper TTS
- **Pros:** 200ms latency, 8MB model size
- **Cons:** Robotic prosody, no voice cloning
- **Use Case:** Fallback if XTTS latency unacceptable

**Architecture:**
```python
from TTS.api import TTS

class TTSEngine:
    def __init__(self):
        # Load XTTS-v2
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        
        # Load reference voice (professional interviewer persona)
        self.reference_audio = "assets/interviewer_voice_sample.wav"
        
    async def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        """
        Convert text to speech audio
        
        Args:
            text: Question or response text
            emotion: "neutral"|"friendly"|"encouraging" (affects prosody)
        
        Returns:
            WAV audio bytes (24kHz, mono)
        """
        # Add prosody hints based on emotion
        if emotion == "encouraging":
            text = f"<speaking_rate=0.95><pitch=+5%>{text}</pitch></speaking_rate>"
        elif emotion == "friendly":
            text = f"<speaking_rate=0.9>{text}</speaking_rate>"
        
        # Generate audio
        wav = self.tts.tts(
            text=text,
            speaker_wav=self.reference_audio,
            language="en"
        )
        
        # Convert to bytes
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        self.tts.synthesizer.save_wav(wav, temp_path)
        
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        
        os.remove(temp_path)
        
        return audio_bytes
```

**Optimization for Low Latency:**
```python
class StreamingTTSEngine:
    """
    Stream audio chunks as they're generated
    Unity can start playing before full synthesis completes
    """
    
    async def synthesize_streaming(self, text: str):
        """
        Yield audio chunks for progressive playback
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            # Generate audio for this sentence
            audio_chunk = await self.synthesize(sentence)
            
            # Yield immediately (Unity starts playing)
            yield audio_chunk
            
            # Small overlap to prevent gaps
            await asyncio.sleep(0.1)
```

**Voice Persona Configuration:**
```python
INTERVIEWER_PERSONAS = {
    "professional": {
        "reference_audio": "assets/voice_professional.wav",
        "speaking_rate": 1.0,
        "pitch_shift": 0,
        "description": "Neutral, clear, corporate tone"
    },
    "friendly": {
        "reference_audio": "assets/voice_friendly.wav",
        "speaking_rate": 0.95,
        "pitch_shift": +3,
        "description": "Warm, encouraging, slightly slower"
    },
    "technical": {
        "reference_audio": "assets/voice_technical.wav",
        "speaking_rate": 0.9,
        "pitch_shift": -2,
        "description": "Deeper, measured pace for technical questions"
    }
}

# Select based on interview stage
def get_tts_config(stage: str) -> dict:
    if stage == "screening":
        return INTERVIEWER_PERSONAS["friendly"]
    elif stage == "technical":
        return INTERVIEWER_PERSONAS["technical"]
    else:
        return INTERVIEWER_PERSONAS["professional"]
```

### 4.3 Audio Processing Utilities

**Format Conversion:**
```python
from pydub import AudioSegment

class AudioProcessor:
    @staticmethod
    def convert_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
        """
        Unity might send WebM/OGG, convert to WAV for Whisper
        """
        audio = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format=source_format
        )
        
        # Whisper expects 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        return wav_buffer.getvalue()
    
    @staticmethod
    def normalize_audio(audio_bytes: bytes) -> bytes:
        """
        Normalize volume (helps with quiet microphones)
        """
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        
        # Target -20 dBFS (standard for speech)
        change_in_dBFS = -20.0 - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)
        
        output = io.BytesIO()
        normalized.export(output, format="wav")
        return output.getvalue()
```

### 4.4 End-to-End Voice Flow

**Unity → Backend → Unity:**

```
1. Unity captures microphone
   └─> AudioClip.GetData() → byte[] (PCM)
   └─> Convert to WAV format
   
2. POST /api/interview/respond-audio
   └─> Body: multipart/form-data (audio.wav)
   
3. Backend STT
   └─> Whisper transcribes → text
   └─> Save to MongoDB (audio URL + transcript)
   
4. Backend Agent Processing
   └─> LLM evaluates answer
   └─> Generates next question text
   
5. Backend TTS
   └─> XTTS synthesizes → audio bytes
   └─> Cache in Redis (key: question_id)
   
6. Response to Unity
   └─> JSON: { "transcript": "...", "audio_base64": "..." }
   OR streaming: SSE chunks
   
7. Unity plays audio
   └─> Base64 decode → AudioClip
   └─> AudioSource.PlayOneShot()
```

**API Endpoint:**
```python
@app.post("/api/interview/respond-audio")
async def handle_audio_response(
    session_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Process candidate's audio response
    Returns agent's next question as audio
    """
    # Read audio bytes
    audio_bytes = await audio.read()
    
    # STT: Audio → Text
    stt_result = await stt_engine.transcribe(audio_bytes)
    candidate_text = stt_result['text']
    
    if not candidate_text:
        return JSONResponse({
            "error": "no_speech_detected",
            "message": "Could you please repeat that?"
        }, status_code=400)
    
    # Save audio + transcript
    audio_url = await storage.upload(
        f"interviews/{session_id}/audio_{uuid.uuid4()}.wav",
        audio_bytes
    )
    
    # Agent processes answer
    agent = await get_agent(session_id)
    evaluation = await agent.evaluate_answer(candidate_text)
    next_question = await agent.get_next_question()
    
    # Log interaction
    await db.interactions.insert_one({
        "session_id": session_id,
        "audio_url": audio_url,
        "transcript": candidate_text,
        "evaluation": evaluation.dict(),
        "timestamp": datetime.utcnow()
    })
    
    # TTS: Generate audio for next question
    question_audio = await tts_engine.synthesize(
        next_question.text,
        emotion="neutral"
    )
    
    # Return both text and audio
    return {
        "transcript": candidate_text,  # Echo back (for UI confirmation)
        "evaluation_score": evaluation.score,  # Optional: show live score
        "next_question": {
            "text": next_question.text,
            "audio_base64": base64.b64encode(question_audio).decode('utf-8'),
            "audio_duration_ms": len(AudioSegment.from_wav(io.BytesIO(question_audio)))
        }
    }
```

### 4.5 Performance Optimization

**Caching Strategy:**
```python
class TTSCache:
    """
    Pre-generate audio for common questions
    80% of screening questions are standard
    """
    
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=1)
        
    async def get_or_generate(self, question_text: str) -> bytes:
        # Check cache
        cache_key = f"tts:{hashlib.sha256(question_text.encode()).hexdigest()}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return cached
        
        # Generate and cache
        audio = await tts_engine.synthesize(question_text)
        self.redis.setex(cache_key, 3600, audio)  # 1hr TTL
        
        return audio

# Pre-warm cache on startup
COMMON_QUESTIONS = [
    "Can you tell me about yourself?",
    "What interests you about this role?",
    "Walk me through your resume.",
    # ... top 50 most common questions
]

@app.on_event("startup")
async def warm_tts_cache():
    for question in COMMON_QUESTIONS:
        await tts_cache.get_or_generate(question)
```

**Parallel Processing:**
```python
async def process_response_parallel(session_id: str, audio_bytes: bytes):
    """
    Run STT and next question generation in parallel
    Saves ~500ms per turn
    """
    # Start STT immediately
    stt_task = asyncio.create_task(stt_engine.transcribe(audio_bytes))
    
    # While STT runs, prepare next question context
    agent = await get_agent(session_id)
    context_task = asyncio.create_task(agent.prepare_next_context())
    
    # Wait for STT
    transcript = await stt_task
    
    # Wait for context (should be done by now)
    context = await context_task
    
    # Now evaluate answer and generate next question
    # (can't parallelize this as evaluation depends on transcript)
    evaluation = await agent.evaluate_answer(transcript['text'])
    next_question = await agent.get_next_question(evaluation, context)
    
    # Generate TTS
    audio = await tts_engine.synthesize(next_question.text)
    
    return transcript, evaluation, next_question, audio
```

### 4.6 Quality Assurance for Voice

**STT Confidence Scoring:**
```python
def check_transcription_quality(stt_result: dict) -> dict:
    """
    Detect when transcription might be unreliable
    """
    text = stt_result['text']
    
    # Red flags
    flags = []
    
    if len(text.split()) < 3:
        flags.append("too_short")
    
    if text.count('[inaudible]') > 0:
        flags.append("audio_quality_poor")
    
    # Check for repeated words (audio glitch indicator)
    words = text.split()
    if any(words.count(w) > 3 for w in set(words)):
        flags.append("possible_audio_loop")
    
    # Low confidence (if using Whisper with word-level timestamps)
    # avg_confidence = mean([segment['confidence'] for segment in stt_result['segments']])
    # if avg_confidence < 0.7:
    #     flags.append("low_confidence")
    
    return {
        "is_reliable": len(flags) == 0,
        "flags": flags,
        "recommendation": "ask_repeat" if flags else "proceed"
    }
```

**TTS Quality Check:**
```python
async def validate_tts_output(audio_bytes: bytes, expected_text: str) -> bool:
    """
    Verify TTS generated correctly (detect truncation/errors)
    """
    # Check duration matches text length (rough heuristic)
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    duration_sec = len(audio) / 1000.0
    
    # Average English: ~150 words/min = 2.5 words/sec
    expected_duration = len(expected_text.split()) / 2.5
    
    # Allow 30% margin
    if abs(duration_sec - expected_duration) / expected_duration > 0.3:
        logger.warning(f"TTS duration mismatch: {duration_sec}s vs expected {expected_duration}s")
        return False
    
    return True
```

---

## 5. Agent Design

### 5.1 Interview Agent Architecture

**Agent Type:** ReAct (Reasoning + Acting) pattern
- **Reasoning:** LLM decides next question based on candidate performance
- **Acting:** Executes question, evaluates answer, updates state

**Core Agent Loop:**
```python
class InterviewAgent:
    def __init__(self, resume: dict, job_desc: dict):
        self.resume = resume
        self.jd = job_desc
        self.state = "initialization"
        self.question_bank = []
        self.transcript = []
        self.scores = {"technical": [], "behavioral": [], "communication": []}
        
    async def initialize(self):
        """Generate personalized interview plan"""
        # 1. Semantic matching
        match_analysis = self.analyze_fit(self.resume, self.jd)
        
        # 2. Generate question bank
        self.question_bank = await self.generate_questions(match_analysis)
        
        # 3. Set difficulty adaptation thresholds
        self.difficulty_threshold = self._calculate_baseline_difficulty()
        
    async def conduct_interview(self):
        """Main interview loop"""
        stages = ["screening", "technical", "behavioral", "wrapup"]
        
        for stage in stages:
            self.state = stage
            questions = self._get_questions_for_stage(stage)
            
            for question in questions:
                # Ask question (TTS)
                audio = await tts_engine.synthesize(question.text)
                yield {"type": "question", "text": question.text, "audio": audio}
                
                # Wait for candidate response (from Unity via STT)
                audio_response = await self.wait_for_audio_response()
                transcript = await stt_engine.transcribe(audio_response)
                
                # Evaluate answer
                score = await self.evaluate_answer(question, transcript['text'])
                self.scores[question.category].append(score)
                
                # Communication quality check
                comm_score = self.analyze_communication(transcript)
                self.scores["communication"].append(comm_score)
                
                # Adaptive follow-up
                if score.value < 3 and question.allow_followup:
                    followup = await self.generate_followup(question, transcript, score)
                    followup_audio = await tts_engine.synthesize(followup.text)
                    yield {"type": "followup", "text": followup.text, "audio": followup_audio}
                
                # Update transcript
                self.transcript.append({
                    "question": question.text,
                    "answer": transcript['text'],
                    "audio_url": await self.save_audio(audio_response),
                    "score": score.dict(),
                    "timestamp": datetime.utcnow()
                })
                
                # Early termination check (screening stage)
                if stage == "screening" and self._should_terminate_early():
                    termination_msg = "Thank you for your time. We'll be in touch regarding next steps."
                    term_audio = await tts_engine.synthesize(termination_msg)
                    yield {"type": "termination", "text": termination_msg, "audio": term_audio}
                    return
        
        # Generate report
        report = await self.generate_report()
        yield {"type": "report", "content": report}
```

### 5.2 Resume & JD Processing

**Document Parser:**
```python
import spacy
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')  # 1024-dim, MIT license
        self.skill_taxonomy = self._load_skill_taxonomy()
    
    def parse_resume(self, pdf_bytes: bytes) -> dict:
        """
        Extract structured data from resume PDF
        """
        # Extract text
        text = self._extract_text_from_pdf(pdf_bytes)
        
        # NER extraction
        doc = self.nlp(text)
        
        # Skills extraction (custom NER + taxonomy matching)
        skills = self._extract_skills(doc, text)
        
        # Experience extraction
        experience = self._extract_experience(doc, text)
        
        # Education extraction
        education = self._extract_education(doc)
        
        return {
            "raw_text": text,
            "skills": skills,
            "experience": experience,
            "education": education,
            "summary_embedding": self.embedder.encode(text[:1000])  # First 1000 chars
        }
    
    def _extract_skills(self, doc, text: str) -> List[dict]:
        """
        Extract skills using:
        1. spaCy NER (custom trained on tech resumes)
        2. Fuzzy matching against skill taxonomy
        """
        skills = []
        
        # Method 1: NER-based
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skills.append({
                    "name": ent.text,
                    "method": "ner",
                    "context": doc[max(0, ent.start-5):min(len(doc), ent.end+5)].text
                })
        
        # Method 2: Taxonomy matching
        text_lower = text.lower()
        for skill_category, skill_list in self.skill_taxonomy.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills.append({
                        "name": skill,
                        "category": skill_category,
                        "method": "taxonomy"
                    })
        
        # Deduplicate and normalize
        skills = self._deduplicate_skills(skills)
        
        return skills
    
    def parse_job_description(self, jd_json: dict) -> dict:
        """
        Structure job description requirements
        """
        return {
            "role": jd_json.get("title"),
            "seniority_level": jd_json.get("seniority", "mid"),  # junior|mid|senior|staff
            "required_skills": jd_json.get("required_skills", []),
            "preferred_skills": jd_json.get("preferred_skills", []),
            "responsibilities": jd_json.get("responsibilities", []),
            "years_required": jd_json.get("years_experience", 3),
            "core_tech_stack": jd_json.get("tech_stack", []),
            "company_culture": jd_json.get("culture_values", []),
            "jd_embedding": self.embedder.encode(jd_json.get("description", ""))
        }
```

**Semantic Matching:**
```python
class SemanticMatcher:
    def __init__(self):
        self.embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')  # 1024-dim, MIT license
    
    def analyze_fit(self, resume: dict, jd: dict) -> dict:
        """
        Calculate candidate-job fit across multiple dimensions
        """
        # 1. Skill overlap
        skill_match = self._calculate_skill_overlap(
            resume['skills'],
            jd['required_skills'],
            jd['preferred_skills']
        )
        
        # 2. Experience level match
        exp_match = self._calculate_experience_match(
            resume['experience']['total_years'],
            jd['years_required']
        )
        
        # 3. Semantic similarity (overall)
        semantic_sim = cosine_similarity(
            [resume['summary_embedding']],
            [jd['jd_embedding']]
        )[0][0]
        
        # 4. Gap analysis
        gaps = self._identify_gaps(resume['skills'], jd['required_skills'])
        
        # 5. Composite score
        overall_score = (
            0.5 * skill_match['score'] +
            0.2 * exp_match['score'] +
            0.3 * semantic_sim * 100
        )
        
        return {
            "overall_score": overall_score,  # 0-100
            "skill_match": skill_match,
            "experience_match": exp_match,
            "semantic_similarity": semantic_sim,
            "gaps": gaps,
            "recommendation": self._get_recommendation(overall_score)
        }
    
    def _calculate_skill_overlap(self, resume_skills, required, preferred) -> dict:
        """
        Weighted skill matching
        """
        resume_skill_names = {s['name'].lower() for s in resume_skills}
        required_lower = {s.lower() for s in required}
        preferred_lower = {s.lower() for s in preferred}
        
        # Required skills matched
        required_matched = resume_skill_names & required_lower
        required_coverage = len(required_matched) / len(required_lower) if required_lower else 0
        
        # Preferred skills matched (bonus)
        preferred_matched = resume_skill_names & preferred_lower
        preferred_coverage = len(preferred_matched) / len(preferred_lower) if preferred_lower else 0
        
        # Score: 70% weight on required, 30% on preferred
        score = (0.7 * required_coverage + 0.3 * preferred_coverage) * 100
        
        return {
            "score": score,
            "required_matched": list(required_matched),
            "required_missing": list(required_lower - required_matched),
            "preferred_matched": list(preferred_matched)
        }
    
    def _identify_gaps(self, resume_skills, required_skills) -> List[dict]:
        """
        Find skill gaps that should be probed in interview
        """
        resume_skill_names = {s['name'].lower() for s in resume_skills}
        required_lower = {s.lower() for s in required_skills}
        
        missing = required_lower - resume_skill_names
        
        return [
            {
                "skill": skill,
                "severity": "critical",  # All required skills are critical
                "probe_strategy": "direct_question"  # vs "scenario_based"
            }
            for skill in missing
        ]
```

### 5.3 Question Generation

**Technical Question Generator:**
```python
async def generate_technical_questions(self, jd: dict, resume: dict, gaps: list) -> List[Question]:
    """
    Generate 6-10 technical questions based on role requirements
    """
    
    prompt = f"""You are an expert technical interviewer for a {jd['role']} position at {jd['seniority_level']} level.

JOB REQUIREMENTS:
- Core Tech Stack: {', '.join(jd['core_tech_stack'])}
- Required Skills: {', '.join(jd['required_skills'])}
- Key Responsibilities: {chr(10).join(f'  - {r}' for r in jd['responsibilities'][:5])}

CANDIDATE BACKGROUND:
- Skills: {', '.join([s['name'] for s in resume['skills'][:15]])}
- Experience: {resume['experience']['total_years']} years
- Previous Roles: {', '.join([e['title'] for e in resume['experience']['positions'][:3]])}

IDENTIFIED SKILL GAPS TO PROBE:
{chr(10).join(f'  - {g["skill"]}' for g in gaps[:5])}

INSTRUCTIONS:
Generate exactly 8 technical questions following this distribution:

1. FOUNDATIONAL (2 questions) - Core concepts from required tech stack
   - Difficulty: 2-3/5
   - Example: "Explain the difference between {concept_A} and {concept_B} in {technology}"

2. APPLIED (3 questions) - Practical problem-solving in their domain
   - Difficulty: 3-4/5
   - Example: "How would you optimize {specific_scenario} in {technology}?"

3. GAP PROBING (2 questions) - Directly address missing skills
   - Difficulty: 3/5
   - Example: "Walk me through how you'd implement {gap_skill} in a production setting"

4. SYSTEM DESIGN (1 question, only if senior+ role) - Architecture/scalability
   - Difficulty: 4-5/5
   - Example: "Design a system for {use_case} handling {scale} users"

For EACH question, provide this JSON structure:
{{
  "question_id": "tech_001",
  "question_text": "Clear, specific question",
  "category": "technical",
  "subcategory": "foundational|applied|gap_probing|system_design",
  "difficulty": 1-5,
  "time_limit_minutes": 3-8,
  "key_points": [
    "Expected point 1 in answer",
    "Expected point 2 in answer",
    "Expected point 3 in answer",
    "...(5-7 total)"
  ],
  "rubric": {{
    "score_5": "Mentions all key points + demonstrates deep understanding + provides examples",
    "score_4": "Mentions 4-5 key points + shows good understanding",
    "score_3": "Mentions 3 key points + shows basic understanding",
    "score_2": "Mentions 1-2 key points + has conceptual gaps",
    "score_1": "Incorrect understanding or off-topic"
  }},
  "followup_strategy": "If score < 3, ask for specific example or clarification on {weakest_key_point}",
  "tags": ["python", "databases", "optimization"]  # Relevant tech tags
}}

OUTPUT: JSON array of 8 question objects. NO preamble, ONLY valid JSON."""

    response = await self.llm.generate(prompt, max_tokens=3000, temperature=0.7)
    
    # Clean and parse
    json_text = self._extract_json(response)
    questions_data = json.loads(json_text)
    
    # Validate structure
    for q in questions_data:
        assert all(k in q for k in ["question_text", "rubric", "key_points"]), "Missing required fields"
    
    return [Question(**q) for q in questions_data]
```

**Behavioral Question Generator:**
```python
BEHAVIORAL_QUESTION_BANK = {
    "teamwork": [
        {
            "text": "Tell me about a time you had to collaborate with someone whose working style was very different from yours.",
            "star_focus": "Action - how they adapted their approach",
            "culture_fit_signal": "Flexibility, empathy"
        },
        # ... 5 more teamwork questions
    ],
    "leadership": [
        {
            "text": "Describe a situation where you had to influence a team without having formal authority.",
            "star_focus": "Action - persuasion tactics used",
            "culture_fit_signal": "Initiative, emotional intelligence"
        },
        # ... 5 more leadership questions
    ],
    "conflict_resolution": [...],
    "adaptability": [...],
    "problem_solving": [...]
}

async def generate_behavioral_questions(self, jd: dict, resume: dict) -> List[Question]:
    """
    Select 5 behavioral questions based on role requirements and company culture
    """
    
    # Extract soft skill requirements from JD
    required_traits = []
    
    # Parse from culture values
    if "collaborative" in jd.get('company_culture', '').lower():
        required_traits.append("teamwork")
    if "fast-paced" in jd.get('company_culture', '').lower():
        required_traits.append("adaptability")
    if jd['seniority_level'] in ['senior', 'staff', 'principal']:
        required_traits.append("leadership")
    
    # Always include
    required_traits.extend(["problem_solving", "conflict_resolution"])
    
    # Deduplicate and limit to 5
    required_traits = list(dict.fromkeys(required_traits))[:5]
    
    selected_questions = []
    
    for trait in required_traits:
        # Randomly pick one from bank for variety
        question_template = random.choice(BEHAVIORAL_QUESTION_BANK[trait])
        
        # Optionally personalize using LLM
        personalized = await self._personalize_behavioral_question(
            question_template,
            jd,
            resume
        )
        
        selected_questions.append(Question(
            question_id=f"beh_{trait[:3]}_{random.randint(100,999)}",
            question_text=personalized,
            category="behavioral",
            subcategory=trait,
            difficulty=3,  # Behavioral questions are medium difficulty
            time_limit_minutes=4,
            rubric=self._get_star_rubric(),
            star_focus=question_template['star_focus'],
            culture_fit_signal=question_template['culture_fit_signal']
        ))
    
    return selected_questions

def _get_star_rubric(self) -> dict:
    """Standard STAR evaluation rubric"""
    return {
        "score_5": "Complete STAR (all 4 elements clear) + high self-awareness + quantified results",
        "score_4": "Complete STAR + good detail + shows learning",
        "score_3": "Partial STAR (3/4 elements) + adequate specificity",
        "score_2": "Incomplete STAR (2/4 elements) + vague or generic",
        "score_1": "No STAR structure + red flags (blame, lack of ownership)"
    }
```

### 5.4 Adaptive Difficulty System

```python
class DifficultyAdapter:
    """
    Adjusts question difficulty based on real-time performance
    Similar to CAT (Computerized Adaptive Testing)
    """
    
    def __init__(self, question_bank: List[Question]):
        self.question_bank_by_difficulty = self._group_by_difficulty(question_bank)
        self.performance_history = []
        self.current_difficulty = 3  # Start at medium
        
    def get_next_question(self, last_score: Optional[int] = None) -> Question:
        """
        Select next question based on performance trajectory
        """
        if last_score is not None:
            self.performance_history.append(last_score)
            
            # Adjust difficulty
            if len(self.performance_history) >= 2:
                recent_avg = statistics.mean(self.performance_history[-2:])
                
                if recent_avg >= 4.5:
                    self.current_difficulty = min(5, self.current_difficulty + 1)
                elif recent_avg <= 2.0:
                    self.current_difficulty = max(2, self.current_difficulty - 1)
        
        # Get questions at current difficulty
        available = self.question_bank_by_difficulty.get(self.current_difficulty, [])
        
        if not available:
            # Fallback to closest difficulty
            available = self._get_closest_difficulty_questions(self.current_difficulty)
        
        # Random selection from available (prevents pattern recognition)
        return random.choice(available)
    
    def _group_by_difficulty(self, questions: List[Question]) -> dict:
        grouped = {1: [], 2: [], 3: [], 4: [], 5: []}
        for q in questions:
            grouped[q.difficulty].append(q)
        return grouped
```

---

## 6. Assessment Engine

### 6.1 LLM-as-Judge Evaluation

**Technical Answer Evaluator:**
```python
async def evaluate_technical_answer(
    self,
    question: Question,
    answer: str,
    audio_metadata: dict
) -> Score:
    """
    Multi-faceted evaluation of technical answers
    """
    
    # Primary evaluation: Content accuracy
    content_eval = await self._evaluate_content(question, answer)
    
    # Secondary: Communication quality
    comm_eval = self._evaluate_communication(answer, audio_metadata)
    
    # Combine scores (80% content, 20% communication)
    final_score = int(0.8 * content_eval['score'] + 0.2 * comm_eval['score'])
    
    return Score(
        value=final_score,
        category="technical",
        content_evaluation=content_eval,
        communication_evaluation=comm_eval,
        timestamp=datetime.utcnow()
    )

async def _evaluate_content(self, question: Question, answer: str) -> dict:
    """
    LLM-based content evaluation grounded in rubric
    """
    
    evaluation_prompt = f"""You are an expert technical interviewer. Evaluate this candidate's answer.

QUESTION:
{question.question_text}

EXPECTED KEY POINTS (candidate should cover these):
{chr(10).join(f'{i+1}. {point}' for i, point in enumerate(question.key_points))}

CANDIDATE'S ANSWER:
{answer}

EVALUATION TASK:
1. Identify which key points the candidate mentioned (quote exact phrases)
2. Check for technical accuracy (flag any errors or misconceptions)
3. Assess depth of understanding (superficial vs deep knowledge)
4. Note any additional insights beyond key points

SCORING (use the rubric):
{json.dumps(question.rubric, indent=2)}

OUTPUT FORMAT (JSON only, no preamble):
{{
  "score": 1-5,
  "covered_points": [
    {{"point": "Key point 1", "quote": "exact quote from answer", "quality": "excellent|good|partial"}},
    ...
  ],
  "missed_points": ["Key point 4", "Key point 6"],
  "technical_errors": [
    {{"error": "description", "severity": "critical|minor"}}
  ],
  "additional_insights": ["Candidate mentioned X which shows deep understanding"],
  "strengths": ["2-3 bullet points"],
  "weaknesses": ["2-3 bullet points"],
  "justification": "1-2 sentences explaining the score"
}}

CRITICAL RULES:
- Base score ONLY on rubric, not on gut feeling
- If candidate missed 50%+ key points, score cannot exceed 3
- Technical errors automatically cap score at 3 (critical errors) or 4 (minor errors)
- Empty or off-topic answers must score 1"""

    response = await self.llm.generate(
        evaluation_prompt,
        max_tokens=800,
        temperature=0.3  # Lower temp for consistent grading
    )
    
    # Parse and validate
    evaluation = json.loads(self._extract_json(response))
    
    # Post-validation: Check score consistency
    evaluation = self._validate_score_consistency(evaluation, question)
    
    return evaluation

def _validate_score_consistency(self, evaluation: dict, question: Question) -> dict:
    """
    Prevent LLM from being too generous/harsh
    Enforce rules that score must match evidence
    """
    covered_ratio = len(evaluation['covered_points']) / len(question.key_points)
    has_critical_errors = any(
        e['severity'] == 'critical' for e in evaluation.get('technical_errors', [])
    )
    
    # Rule 1: Coverage ratio should correlate with score
    if covered_ratio < 0.5 and evaluation['score'] > 3:
        evaluation['score'] = 3
        evaluation['validation_note'] = "Auto-adjusted: insufficient key point coverage"
    
    # Rule 2: Critical errors cap at score 3
    if has_critical_errors and evaluation['score'] > 3:
        evaluation['score'] = 3
        evaluation['validation_note'] = "Auto-adjusted: critical technical errors present"
    
    # Rule 3: Excellent coverage (90%+) should score at least 4
    if covered_ratio >= 0.9 and not has_critical_errors and evaluation['score'] < 4:
        evaluation['score'] = 4
        evaluation['validation_note'] = "Auto-adjusted: excellent coverage deserves higher score"
    
    return evaluation
```

**Behavioral Answer Evaluator (STAR Analysis):**
```python
async def evaluate_behavioral_answer(
    self,
    question: Question,
    answer: str
) -> Score:
    """
    STAR-method focused evaluation
    """
    
    star_prompt = f"""Evaluate this behavioral interview answer using the STAR framework.

QUESTION:
{question.question_text}

CANDIDATE'S ANSWER:
{answer}

STAR ANALYSIS:
For each component, determine if it's present and extract the relevant quote:

1. SITUATION: Context and background
   - Present? (Yes/No)
   - Quote: "exact text showing situation"
   - Quality: (clear|vague|missing)

2. TASK: Candidate's specific responsibility
   - Present? (Yes/No)
   - Quote: "exact text showing task"
   - Quality: (clear|vague|missing)

3. ACTION: Specific steps taken
   - Present? (Yes/No)
   - Quote: "exact text showing actions"
   - Quality: (specific|generic|missing)
   - Ownership level: (high|medium|low) - did they take initiative?

4. RESULT: Outcomes and learnings
   - Present? (Yes/No)
   - Quote: "exact text showing results"
   - Quality: (quantified|qualitative|missing)
   - Self-awareness: (high|medium|low) - do they reflect?

ADDITIONAL FACTORS:
- Specificity: Are examples concrete or vague?
- Red flags: Blaming others, lack of ownership, ethical concerns
- Culture fit signals: {question.culture_fit_signal}

OUTPUT (JSON):
{{
  "score": 1-5,
  "star_analysis": {{
    "situation": {{"present": true/false, "quote": "...", "quality": "clear|vague|missing"}},
    "task": {{...}},
    "action": {{...}},
    "result": {{...}}
  }},
  "star_completeness": "4/4" or "3/4" etc.,
  "specificity_level": "high|medium|low",
  "self_awareness": "high|medium|low",
  "ownership_level": "high|medium|low",
  "red_flags": [],
  "culture_fit_alignment": "strong|moderate|weak",
  "justification": "..."
}}

SCORING RUBRIC:
5: Complete STAR + high specificity + strong self-awareness + quantified results
4: Complete STAR + good detail + shows learning
3: 3/4 STAR elements + adequate specificity
2: 2/4 STAR elements + vague
1: <2 STAR elements OR red flags present"""

    response = await self.llm.generate(star_prompt, max_tokens=700, temperature=0.3)
    evaluation = json.loads(self._extract_json(response))
    
    # Validation: STAR completeness should match score
    star_count = sum(1 for k, v in evaluation['star_analysis'].items() if v['present'])
    
    if star_count >= 4 and evaluation['score'] < 3:
        evaluation['score'] = max(3, evaluation['score'])
    elif star_count <= 1 and evaluation['score'] > 2:
        evaluation['score'] = min(2, evaluation['score'])
    
    return Score(
        value=evaluation['score'],
        category="behavioral",
        subcategory=question.subcategory,
        details=evaluation
    )
```

### 6.2 Communication Quality Analysis

```python
from textblob import TextBlob
import language_tool_python

class CommunicationAnalyzer:
    def __init__(self):
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
    
    def analyze(self, answer_text: str, audio_metadata: dict) -> dict:
        """
        Automated communication quality metrics
        """
        return {
            # Structural metrics
            "word_count": len(answer_text.split()),
            "sentence_count": len(nltk.sent_tokenize(answer_text)),
            "avg_sentence_length": self._avg_sentence_length(answer_text),
            
            # Clarity metrics
            "flesch_reading_ease": self._flesch_score(answer_text),  # 0-100, higher = clearer
            "grammar_errors": len(self.grammar_tool.check(answer_text)),
            
            # Confidence indicators
            "filler_word_count": self._count_fillers(answer_text),
            "hedge_word_count": self._count_hedges(answer_text),  # "maybe", "I think"
            "confidence_score": self._calculate_confidence(answer_text),
            
            # Audio-based (from STT metadata)
            "speaking_rate_wpm": audio_metadata.get('speaking_rate', 0),
            "pause_count": audio_metadata.get('pause_count', 0),
            
            # Overall score (1-5)
            "communication_score": self._aggregate_comm_score(answer_text, audio_metadata)
        }
    
    def _count_fillers(self, text: str) -> int:
        fillers = ["um", "uh", "like", "you know", "sort of", "kind of", "actually"]
        text_lower = text.lower()
        return sum(text_lower.count(f" {filler} ") for filler in fillers)
    
    def _count_hedges(self, text: str) -> int:
        hedges = ["i think", "maybe", "probably", "i guess", "perhaps", "somewhat"]
        text_lower = text.lower()
        return sum(text_lower.count(hedge) for hedge in hedges)
    
    def _calculate_confidence(self, text: str) -> float:
        """
        Confidence score based on linguistic markers
        0.0 = very hesitant, 1.0 = very confident
        """
        # Positive indicators
        assertive_words = ["definitely", "clearly", "certainly", "absolutely"]
        assertive_count = sum(text.lower().count(word) for word in assertive_words)
        
        # Negative indicators
        filler_count = self._count_fillers(text)
        hedge_count = self._count_hedges(text)
        
        # Normalize by word count
        word_count = len(text.split())
        if word_count == 0:
            return 0.5
        
        confidence = 0.5  # Baseline
        confidence += (assertive_count / word_count) * 10  # Boost for assertive words
        confidence -= (filler_count / word_count) * 15  # Penalty for fillers
        confidence -= (hedge_count / word_count) * 10  # Penalty for hedges
        
        return max(0.0, min(1.0, confidence))
    
    def _aggregate_comm_score(self, text: str, audio_metadata: dict) -> int:
        """
        Convert metrics to 1-5 score
        """
        score = 3.0  # Start at average
        
        # Grammar (±1 point)
        grammar_errors = len(self.grammar_tool.check(text))
        if grammar_errors == 0:
            score += 1
        elif grammar_errors > 5:
            score -= 1
        
        # Clarity (±0.5 points)
        flesch = self._flesch_score(text)
        if flesch > 60:  # Easy to read
            score += 0.5
        elif flesch < 30:  # Difficult
            score -= 0.5
        
        # Confidence (±0.5 points)
        confidence = self._calculate_confidence(text)
        if confidence > 0.7:
            score += 0.5
        elif confidence < 0.3:
            score -= 0.5
        
        return max(1, min(5, int(round(score))))
```

---

## 7. Report Generation

### 7.1 Score Aggregation

```python
class ReportGenerator:
    def __init__(self, interview_data: dict):
        self.data = interview_data
        self.transcript = interview_data['transcript']
        self.scores = interview_data['scores']
    
    async def generate_comprehensive_report(self) -> dict:
        """
        Generate hiring report with multiple sections
        """
        # Calculate aggregate scores
        technical_score = self._calculate_technical_score()
        behavioral_score = self._calculate_behavioral_score()
        communication_score = self._calculate_communication_score()
        overall_score = self._calculate_overall_score(
            technical_score,
            behavioral_score,
            communication_score
        )
        
        # Generate insights
        strengths = await self._identify_strengths()
        weaknesses = await self._identify_weaknesses()
        red_flags = self._extract_red_flags()
        
        # Hiring recommendation
        recommendation = self._generate_recommendation(overall_score, red_flags)
        
        return {
            "candidate_info": {
                "name": self.data['candidate_name'],
                "role_applied": self.data['job_description']['role'],
                "interview_date": self.data['interview_date'],
                "duration_minutes": self.data['duration_minutes']
            },
            "scores": {
                "overall": overall_score,
                "technical": technical_score,
                "behavioral": behavioral_score,
                "communication": communication_score,
                "breakdown": self._get_score_breakdown()
            },
            "analysis": {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "red_flags": red_flags,
                "skill_gaps": self.data['gaps']
            },
            "recommendation": recommendation,
            "transcript_summary": await self._generate_transcript_summary(),
            "next_steps": self._suggest_next_steps(recommendation),
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "model_version": "llama-3.1-8b",
                "report_id": str(uuid.uuid4())
            }
        }
    
    def _calculate_technical_score(self) -> dict:
        """
        Weighted average of technical question scores
        Higher difficulty questions weighted more
        """
        technical_scores = [
            s for s in self.scores['technical']
            if s.category == 'technical'
        ]
        
        if not technical_scores:
            return {"score": 0, "confidence": "low"}
        
        # Weight by difficulty
        weighted_sum = sum(s.value * s.difficulty for s in technical_scores)
        total_weight = sum(s.difficulty for s in technical_scores)
        
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
        
        return {
            "score": round(weighted_avg, 1),
            "out_of": 5,
            "questions_answered": len(technical_scores),
            "confidence": "high" if len(technical_scores) >= 5 else "medium"
        }
    
    def _calculate_behavioral_score(self) -> dict:
        """
        Average behavioral scores with STAR completeness weighting
        """
        behavioral_scores = self.scores['behavioral']
        
        if not behavioral_scores:
            return {"score": 0, "confidence": "low"}
        
        # Weight by STAR completeness
        weighted_scores = []
        for score in behavioral_scores:
            star_count = int(score.details['star_completeness'].split('/')[0])
            star_weight = star_count / 4.0  # 0.25, 0.5, 0.75, 1.0
            weighted_scores.append(score.value * (0.7 + 0.3 * star_weight))
        
        avg = statistics.mean(weighted_scores)
        
        return {
            "score": round(avg, 1),
            "out_of": 5,
            "avg_star_completeness": statistics.mean([
                int(s.details['star_completeness'].split('/')[0])
                for s in behavioral_scores
            ]),
            "confidence": "high"
        }
    
    def _calculate_overall_score(self, tech, behavioral, comm) -> dict:
        """
        Composite score with configurable weights
        """
        # Weights based on role seniority
        seniority = self.data['job_description']['seniority_level']
        
        if seniority in ['junior', 'entry']:
            weights = {"technical": 0.5, "behavioral": 0.3, "communication": 0.2}
        elif seniority in ['senior', 'staff', 'principal']:
            weights = {"technical": 0.4, "behavioral": 0.4, "communication": 0.2}
        else:  # mid-level
            weights = {"technical": 0.45, "behavioral": 0.35, "communication": 0.2}
        
        overall = (
            weights['technical'] * tech['score'] +
            weights['behavioral'] * behavioral['score'] +
            weights['communication'] * comm['score']
        )
        
        return {
            "score": round(overall, 1),
            "out_of": 5,
            "rating": self._score_to_rating(overall),
            "weights_used": weights
        }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 4.5:
            return "Exceptional"
        elif score >= 4.0:
            return "Strong"
        elif score >= 3.0:
            return "Acceptable"
        elif score >= 2.0:
            return "Weak"
        else:
            return "Insufficient"
    
    async def _identify_strengths(self) -> List[str]:
        """
        Use LLM to synthesize strengths from high-scoring areas
        """
        high_scores = [
            s for s in (self.scores['technical'] + self.scores['behavioral'])
            if s.value >= 4
        ]
        
        if not high_scores:
            return ["No significant strengths identified"]
        
        # Aggregate evidence
        evidence = []
        for score in high_scores:
            evidence.append({
                "question": score.question_text,
                "score": score.value,
                "highlights": score.details.get('strengths', [])
            })
        
        prompt = f"""Based on these high-scoring interview responses, identify 3-5 key strengths:

EVIDENCE:
{json.dumps(evidence, indent=2)}

Generate 3-5 bullet points describing candidate's core strengths. Each should be:
- Specific (cite examples from responses)
- Action-oriented (what they demonstrated)
- Relevant to {self.data['job_description']['role']} role

Output as JSON array of strings."""

        response = await self.llm.generate(prompt, max_tokens=400)
        return json.loads(self._extract_json(response))
    
    async def _identify_weaknesses(self) -> List[str]:
        """
        Synthesize weaknesses from low-scoring areas
        """
        low_scores = [
            s for s in (self.scores['technical'] + self.scores['behavioral'])
            if s.value <= 2
        ]
        
        if not low_scores:
            return ["No major weaknesses identified"]
        
        evidence = []
        for score in low_scores:
            evidence.append({
                "question": score.question_text,
                "score": score.value,
                "issues": score.details.get('weaknesses', [])
            })
        
        prompt = f"""Based on these low-scoring responses, identify 3-5 areas for improvement:

EVIDENCE:
{json.dumps(evidence, indent=2)}

Generate 3-5 bullet points. Be constructive and specific. Each should mention:
- What was lacking
- Impact on role performance
- Suggestion for improvement

Output as JSON array of strings."""

        response = await self.llm.generate(prompt, max_tokens=400)
        return json.loads(self._extract_json(response))
    
    def _extract_red_flags(self) -> List[dict]:
        """
        Collect serious concerns from evaluations
        """
        red_flags = []
        
        for score in self.scores['behavioral']:
            if 'red_flags' in score.details and score.details['red_flags']:
                for flag in score.details['red_flags']:
                    red_flags.append({
                        "type": "behavioral",
                        "severity": "high",
                        "description": flag,
                        "context": score.question_text
                    })
        
        for score in self.scores['technical']:
            if 'technical_errors' in score.details:
                critical_errors = [
                    e for e in score.details['technical_errors']
                    if e.get('severity') == 'critical'
                ]
                for error in critical_errors:
                    red_flags.append({
                        "type": "technical",
                        "severity": "medium",
                        "description": error['error'],
                        "context": score.question_text
                    })
        
        return red_flags
    
    def _generate_recommendation(self, overall_score: dict, red_flags: List) -> dict:
        """
        Generate hiring decision recommendation
        """
        score = overall_score['score']
        has_critical_flags = any(f['severity'] == 'high' for f in red_flags)
        
        if has_critical_flags:
            decision = "Do Not Hire"
            confidence = "High"
            reasoning = "Critical red flags identified that pose risk to team/company."
        elif score >= 4.0:
            decision = "Strong Hire"
            confidence = "High"
            reasoning = "Candidate demonstrates strong technical and behavioral competencies aligned with role requirements."
        elif score >= 3.5:
            decision = "Hire"
            confidence = "Medium"
            reasoning = "Candidate meets requirements with minor gaps that can be addressed through onboarding."
        elif score >= 3.0:
            decision = "Borderline - Further Assessment"
            confidence = "Medium"
            reasoning = "Candidate shows potential but has notable gaps. Recommend additional round or technical assessment."
        else:
            decision = "Do Not Hire"
            confidence = "High"
            reasoning = "Candidate does not meet minimum requirements for the role."
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_factors": red_flags,
            "overall_score": overall_score
        }
    
    async def _generate_transcript_summary(self) -> str:
        """
        LLM-generated executive summary of interview
        """
        # Take first/last turn + highest/lowest scoring turns
        key_turns = self._select_key_turns(5)
        
        prompt = f"""Generate a 3-4 sentence executive summary of this interview:

ROLE: {self.data['job_description']['role']}
OVERALL SCORE: {self.data['overall_score']}/5

KEY MOMENTS:
{chr(10).join([f"Q: {t['question']}\nA: {t['answer'][:200]}..." for t in key_turns])}

Summary should highlight:
1. Candidate's overall performance
2. Notable strengths
3. Key concerns (if any)

Keep professional tone, 3-4 sentences max."""

        response = await self.llm.generate(prompt, max_tokens=200)
        return response.strip()
    
    def _suggest_next_steps(self, recommendation: dict) -> List[str]:
        """
        Suggest action items based on recommendation
        """
        decision = recommendation['decision']
        
        next_steps_map = {
            "Strong Hire": [
                "Extend offer immediately",
                "Prepare onboarding materials",
                "Schedule team introduction call"
            ],
            "Hire": [
                "Extend offer with standard timeline",
                "Prepare onboarding plan addressing identified gaps",
                "Consider mentorship pairing for weak areas"
            ],
            "Borderline - Further Assessment": [
                "Schedule follow-up technical assessment",
                "Arrange culture-fit interview with team",
                "Request work samples or take-home project"
            ],
            "Do Not Hire": [
                "Send rejection email with feedback (if policy allows)",
                "Keep resume on file for future opportunities",
                "Analyze screening process to improve candidate quality"
            ]
        }
        
        return next_steps_map.get(decision, ["Review with hiring manager"])
```

### 7.2 PDF Report Generation

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

class PDFReportGenerator:
    def __init__(self, report_data: dict):
        self.data = report_data
        self.styles = getSampleStyleSheet()
        self._add_custom_styles()
    
    def _add_custom_styles(self):
        """Add custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate_pdf(self, output_path: str):
        """
        Generate comprehensive PDF report
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Header
        story.append(Paragraph("Interview Assessment Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Candidate Info
        story.extend(self._build_candidate_section())
        story.append(Spacer(1, 0.3*inch))
        
        # Score Summary (with visual table)
        story.extend(self._build_score_summary())
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendation
        story.extend(self._build_recommendation_section())
        story.append(PageBreak())
        
        # Detailed Analysis
        story.extend(self._build_detailed_analysis())
        story.append(PageBreak())
        
        # Question-by-Question Breakdown
        story.extend(self._build_qna_breakdown())
        
        # Build PDF
        doc.build(story)
    
    def _build_candidate_section(self) -> List:
        """Candidate information section"""
        info = self.data['candidate_info']
        
        data = [
            ['Candidate Name:', info['name']],
            ['Position:', info['role_applied']],
            ['Interview Date:', info['interview_date']],
            ['Duration:', f"{info['duration_minutes']} minutes"]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return [table]
    
    def _build_score_summary(self) -> List:
        """Score summary with color-coded table"""
        scores = self.data['scores']
        
        elements = []
        elements.append(Paragraph("Score Summary", self.styles['SectionHeader']))
        
        # Overall score (large)
        overall_text = f"<b>Overall Score: {scores['overall']['score']}/5</b> - {scores['overall']['rating']}"
        elements.append(Paragraph(overall_text, self.styles['Heading3']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Breakdown table
        data = [
            ['Category', 'Score', 'Rating'],
            ['Technical', f"{scores['technical']['score']}/5", self._score_to_rating(scores['technical']['score'])],
            ['Behavioral', f"{scores['behavioral']['score']}/5", self._score_to_rating(scores['behavioral']['score'])],
            ['Communication', f"{scores['communication']['score']}/5", self._score_to_rating(scores['communication']['score'])]
        ]
        
        table = Table(data, colWidths=[2*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        return elements
    
    def _build_recommendation_section(self) -> List:
        """Hiring recommendation section"""
        rec = self.data['recommendation']
        
        elements = []
        elements.append(Paragraph("Hiring Recommendation", self.styles['SectionHeader']))
        
        # Decision (color-coded)
        decision_color = self._get_decision_color(rec['decision'])
        decision_text = f"<b><font color='{decision_color}'>{rec['decision']}</font></b>"
        elements.append(Paragraph(decision_text, self.styles['Heading3']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Reasoning
        elements.append(Paragraph(f"<b>Reasoning:</b> {rec['reasoning']}", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Red flags (if any)
        if rec['risk_factors']:
            elements.append(Paragraph("<b>Risk Factors:</b>", self.styles['Normal']))
            for flag in rec['risk_factors']:
                elements.append(Paragraph(
                    f"• [{flag['severity'].upper()}] {flag['description']}",
                    self.styles['Normal']
                ))
        
        return elements
    
    def _build_detailed_analysis(self) -> List:
        """Strengths and weaknesses"""
        analysis = self.data['analysis']
        
        elements = []
        elements.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))
        
        # Strengths
        elements.append(Paragraph("<b>Strengths:</b>", self.styles['Heading4']))
        for strength in analysis['strengths']:
            elements.append(Paragraph(f"• {strength}", self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Weaknesses
        elements.append(Paragraph("<b>Areas for Improvement:</b>", self.styles['Heading4']))
        for weakness in analysis['weaknesses']:
            elements.append(Paragraph(f"• {weakness}", self.styles['Normal']))
        
        return elements
    
    def _build_qna_breakdown(self) -> List:
        """Question-by-question breakdown"""
        elements = []
        elements.append(Paragraph("Interview Transcript Summary", self.styles['SectionHeader']))
        
        for i, turn in enumerate(self.data['transcript'][:10], 1):  # Limit to 10 for brevity
            elements.append(Paragraph(f"<b>Q{i}:</b> {turn['question']}", self.styles['Normal']))
            elements.append(Paragraph(
                f"<b>Score:</b> {turn['score']['value']}/5",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _get_decision_color(self, decision: str) -> str:
        """Color code decisions"""
        if "Strong Hire" in decision:
            return "#228B22"  # Green
        elif "Hire" in decision:
            return "#32CD32"  # Light green
        elif "Borderline" in decision:
            return "#FFA500"  # Orange
        else:
            return "#DC143C"  # Red

---

## 8. API Specifications

### 8.1 Core Endpoints

**Initialize Interview Session:**
```python
POST /api/v1/interview/initialize

Request Body:
{
  "candidate_name": "John Doe",
  "resume_pdf": "base64_encoded_pdf",
  "job_description": {
    "title": "Senior Software Engineer",
    "seniority": "senior",
    "required_skills": ["Python", "AWS", "Docker"],
    "preferred_skills": ["Kubernetes", "ML"],
    "responsibilities": ["Design scalable systems", "..."],
    "years_experience": 5,
    "company_culture": "Fast-paced, collaborative startup"
  }
}

Response:
{
  "session_id": "uuid-v4",
  "interview_plan": {
    "total_questions": 13,
    "estimated_duration_minutes": 35,
    "stages": ["screening", "technical", "behavioral", "wrapup"],
    "match_score": 78  // Resume-JD fit score
  },
  "status": "initialized"
}
```

**Start Interview:**
```python
POST /api/v1/interview/{session_id}/start

Response:
{
  "first_question": {
    "question_id": "screen_001",
    "text": "Can you tell me about yourself and what interests you in this role?",
    "audio_base64": "...",  // TTS-generated audio
    "expected_duration_seconds": 120,
    "stage": "screening"
  }
}
```

**Submit Answer (Audio):**
```python
POST /api/v1/interview/{session_id}/answer

Content-Type: multipart/form-data

Form Data:
- question_id: "screen_001"
- audio: <binary WAV file>

Response:
{
  "transcript": "I'm a senior software engineer with 6 years...",
  "evaluation": {
    "score": 4,
    "category": "screening",
    "feedback": null  // Not shown to candidate during interview
  },
  "next_question": {
    "question_id": "tech_001",
    "text": "Explain the difference between horizontal and vertical scaling...",
    "audio_base64": "...",
    "stage": "technical"
  },
  "progress": {
    "current_question": 2,
    "total_questions": 13,
    "stage": "technical"
  }
}
```

**Alternative: Submit Answer (Text - for testing):**
```python
POST /api/v1/interview/{session_id}/answer-text

Request Body:
{
  "question_id": "tech_001",
  "answer_text": "Horizontal scaling means adding more servers..."
}

Response: <same as audio endpoint>
```

**WebSocket for Streaming (Advanced):**
```python
WS /api/v1/interview/{session_id}/stream

Client → Server:
{
  "type": "audio_chunk",
  "data": "base64_audio_chunk"
}

Server → Client (during answer):
{
  "type": "transcription_partial",
  "text": "I would approach this by..."
}

Server → Client (after answer):
{
  "type": "evaluation_complete",
  "score": 4,
  "next_question": {...}
}
```

**Get Interview Status:**
```python
GET /api/v1/interview/{session_id}/status

Response:
{
  "session_id": "...",
  "status": "in_progress",  // initialized|in_progress|completed|terminated
  "current_stage": "technical",
  "questions_completed": 7,
  "total_questions": 13,
  "duration_so_far_minutes": 18,
  "current_scores": {
    "technical_avg": 3.8,
    "behavioral_avg": null  // Not started yet
  }
}
```

**Complete Interview & Generate Report:**
```python
POST /api/v1/interview/{session_id}/complete

Response:
{
  "report": {
    "candidate_info": {...},
    "scores": {...},
    "recommendation": {...},
    "strengths": [...],
    "weaknesses": [...]
  },
  "report_pdf_url": "https://s3.../interview_report_uuid.pdf",
  "report_json_url": "https://s3.../interview_report_uuid.json"
}
```

### 8.2 Administrative Endpoints

**Upload Documents to Knowledge Base:**
```python
POST /api/v1/admin/documents/upload

Content-Type: multipart/form-data

Form Data:
- file: <PDF/DOCX>
- category: "policy|faq|job_description"

Response:
{
  "doc_id": "uuid",
  "filename": "employee_handbook.pdf",
  "chunks_created": 87,
  "indexed": true
}
```

**Get Interview Analytics:**
```python
GET /api/v1/admin/analytics?start_date=2026-01-01&end_date=2026-02-01

Response:
{
  "total_interviews": 156,
  "avg_duration_minutes": 32,
  "avg_overall_score": 3.4,
  "hire_rate": 0.38,
  "strong_hire_rate": 0.12,
  "most_common_weaknesses": [
    {"area": "System Design", "frequency": 67},
    {"area": "Communication", "frequency": 45}
  ]
}
```

---

## 9. Data Architecture

### 9.1 MongoDB Collections

**Collection: `interview_sessions`**
```javascript
{
  "_id": ObjectId,
  "session_id": "uuid-v4",
  "candidate_name": "John Doe",
  "job_description": {
    "title": "Senior Software Engineer",
    "seniority": "senior",
    "required_skills": [],
    // ... full JD
  },
  "resume": {
    "s3_url": "https://s3.../resumes/uuid.pdf",
    "parsed_data": {
      "skills": [],
      "experience": {},
      "education": []
    }
  },
  "match_analysis": {
    "overall_score": 78,
    "skill_match": {...},
    "gaps": [...]
  },
  "interview_plan": {
    "questions": [
      {
        "question_id": "tech_001",
        "question_text": "...",
        "category": "technical",
        "difficulty": 4,
        "rubric": {...}
      },
      // ... all generated questions
    ],
    "total_questions": 13
  },
  "status": "in_progress",
  "created_at": ISODate,
  "started_at": ISODate,
  "completed_at": ISODate|null
}

// Indexes
db.interview_sessions.createIndex({"session_id": 1}, {unique: true})
db.interview_sessions.createIndex({"status": 1, "created_at": -1})
```

**Collection: `interview_interactions`**
```javascript
{
  "_id": ObjectId,
  "session_id": "uuid-v4",
  "turn_number": 1,
  "question_id": "tech_001",
  "question_text": "Explain horizontal vs vertical scaling",
  "question_category": "technical",
  "question_difficulty": 4,
  
  // Candidate response
  "answer_text": "Horizontal scaling means adding more servers...",
  "answer_audio_url": "https://s3.../audio/uuid.wav",
  "answer_duration_seconds": 45,
  
  // STT metadata
  "transcription_confidence": 0.94,
  "speaking_rate_wpm": 145,
  
  // Evaluation
  "score": {
    "value": 4,
    "category": "technical",
    "content_evaluation": {
      "covered_points": [...],
      "missed_points": [...],
      "strengths": [...],
      "weaknesses": [...]
    },
    "communication_evaluation": {
      "grammar_errors": 1,
      "filler_count": 2,
      "confidence_score": 0.82
    }
  },
  
  // Timings
  "timestamp": ISODate,
  "processing_latency_ms": {
    "stt": 1420,
    "evaluation": 890,
    "tts": 1650,
    "total": 3960
  }
}

// Indexes
db.interview_interactions.createIndex({"session_id": 1, "turn_number": 1})
db.interview_interactions.createIndex({"timestamp": -1})
```

**Collection: `interview_reports`**
```javascript
{
  "_id": ObjectId,
  "session_id": "uuid-v4",
  "candidate_name": "John Doe",
  "role": "Senior Software Engineer",
  "interview_date": ISODate,
  "duration_minutes": 34,
  
  "scores": {
    "overall": {
      "score": 4.1,
      "rating": "Strong"
    },
    "technical": {"score": 4.3, "questions_answered": 8},
    "behavioral": {"score": 3.9, "questions_answered": 5},
    "communication": {"score": 4.0}
  },
  
  "analysis": {
    "strengths": [...],
    "weaknesses": [...],
    "red_flags": []
  },
  
  "recommendation": {
    "decision": "Strong Hire",
    "confidence": "High",
    "reasoning": "...",
    "next_steps": [...]
  },
  
  "transcript_summary": "Candidate demonstrated strong...",
  
  "report_urls": {
    "pdf": "https://s3.../reports/uuid.pdf",
    "json": "https://s3.../reports/uuid.json"
  },
  
  "generated_at": ISODate,
  "model_version": "llama-3.1-8b"
}

// Indexes
db.interview_reports.createIndex({"session_id": 1}, {unique: true})
db.interview_reports.createIndex({"interview_date": -1})
db.interview_reports.createIndex({"recommendation.decision": 1})
```

---

## 10. Security & Compliance

### 10.1 Data Privacy

**PII Handling:**
```python
class PIIProtection:
    """
    Anonymize candidate data in logs and analytics
    """
    
    @staticmethod
    def anonymize_for_logging(session_data: dict) -> dict:
        """Remove PII from logs"""
        return {
            "session_id": session_data['session_id'],
            "role": session_data['job_description']['title'],
            "candidate_hash": hashlib.sha256(
                session_data['candidate_name'].encode()
            ).hexdigest()[:16],
            # Never log: name, email, phone, address
        }
    
    @staticmethod
    def encrypt_audio_at_rest(audio_bytes: bytes, key: bytes) -> bytes:
        """Encrypt audio files before S3 upload"""
        from cryptography.fernet import Fernet
        f = Fernet(key)
        return f.encrypt(audio_bytes)
```

**Access Control:**
```python
# JWT-based authentication
RBAC_PERMISSIONS = {
    "interviewer": ["start_interview", "view_session"],
    "hiring_manager": ["start_interview", "view_session", "view_report", "download_report"],
    "admin": ["*"]
}

@app.middleware("http")
async def check_permissions(request: Request, call_next):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    claims = verify_jwt(token)
    
    required_permission = ENDPOINT_PERMISSIONS.get(request.url.path)
    if required_permission and required_permission not in claims['permissions']:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    
    return await call_next(request)
```

### 10.2 Audio Data Retention

```python
# Auto-delete audio after 30 days (GDPR compliance)
@app.on_event("startup")
@repeat_every(seconds=86400)  # Daily
async def cleanup_old_audio():
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    old_interactions = await db.interview_interactions.find({
        "timestamp": {"$lt": cutoff_date}
    }).to_list(None)
    
    for interaction in old_interactions:
        # Delete from S3
        await s3_client.delete_object(
            Bucket=AUDIO_BUCKET,
            Key=extract_key_from_url(interaction['answer_audio_url'])
        )
        
        # Remove URL from DB
        await db.interview_interactions.update_one(
            {"_id": interaction['_id']},
            {"$set": {"answer_audio_url": "[DELETED]", "audio_retention_expired": True}}
        )
```

---

## 11. Performance Requirements

### 11.1 Latency Targets

| Operation | Target | Max Acceptable |
|-----------|--------|----------------|
| STT (10s audio) | 1.5s | 3s |
| Answer evaluation | 2s | 4s |
| TTS (50 words) | 2s | 4s |
| **Total turn latency** | **5s** | **8s** |
| Report generation | 20s | 30s |

### 11.2 Optimization Strategies

**Batch Processing:**
```python
class BatchProcessor:
    """
    Process multiple audio chunks in parallel
    """
    
    async def process_batch(self, audio_chunks: List[bytes]) -> List[str]:
        tasks = [stt_engine.transcribe(chunk) for chunk in audio_chunks]
        results = await asyncio.gather(*tasks)
        return [r['text'] for r in results]
```

**Model Caching:**
```python
# Pre-load models into GPU memory on startup
@app.on_event("startup")
async def warmup_models():
    # LLM warmup
    await llm_engine.generate("Hello", max_tokens=10)
    
    # STT warmup
    dummy_audio = generate_silence(duration=1.0)
    await stt_engine.transcribe(dummy_audio)
    
    # TTS warmup
    await tts_engine.synthesize("Test")
    
    logger.info("All models warmed up and ready")
```

---

## 12. Timeline & Milestones

### Week 1: Foundation
- ✅ Setup FastAPI project structure
- ✅ Configure MongoDB + S3
- ✅ Implement resume parser (spaCy NER)
- ✅ Build JD-resume semantic matcher
- **Deliverable:** Resume parsing working with test PDFs

### Week 2-3: LLM Integration
- ✅ Deploy Llama-3.1-8B with vLLM
- ✅ Implement question generation pipeline
- ✅ Build LLM-as-judge evaluator
- ✅ Validate hallucination detection
- **Deliverable:** End-to-end interview flow (text-only, no voice)

### Week 4: Voice Pipeline
- ✅ Integrate Whisper STT
- ✅ Integrate XTTS TTS
- ✅ Build audio processing utilities
- ✅ Implement audio API endpoints
- **Deliverable:** Full voice interview working locally

### Week 5: Agent Logic
- ✅ Implement interview state machine
- ✅ Build adaptive difficulty system
- ✅ Create follow-up question logic
- ✅ Add early termination rules
- **Deliverable:** Intelligent agent that adjusts to candidate

### Week 6: Report Generation
- ✅ Build score aggregation engine
- ✅ Implement PDF report generator
- ✅ Create insights synthesis (LLM-based)
- ✅ Add hiring recommendation logic
- **Deliverable:** Professional PDF reports

### Week 7: Testing & Optimization
- ✅ Performance benchmarking (meet latency targets)
- ✅ Load testing (10 concurrent interviews)
- ✅ End-to-end integration tests
- ✅ Fix bugs from testing
- **Deliverable:** Production-ready system

### Week 8: Documentation & Deployment
- ✅ API documentation (OpenAPI/Swagger)
- ✅ Deployment scripts (Docker + cloud)
- ✅ Setup guide for client
- ✅ Demo preparation
- **Deliverable:** Complete handoff package

---

## 13. Evaluation Metrics

### 13.1 Model Performance

**Question Quality:**
- Human evaluation: 50 generated questions rated by hiring managers
- Target: >90% relevance to JD
- Metric: % of questions deemed "appropriate for role"

**Evaluation Consistency:**
- Inter-rater reliability: LLM scores vs human interviewer scores
- Target: Cohen's kappa >0.75 (substantial agreement)
- Sample: 100 candidate answers scored by both

**Hallucination Rate:**
- Metric: % of evaluations with validation_override flag
- Target: <5% (95% of evaluations are grounded)

### 13.2 System Performance

**Latency (p95):**
```python
# Monitoring code
@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    
    await metrics.record_latency(
        endpoint=request.url.path,
        latency_ms=latency
    )
    
    return response

# Grafana query
SELECT percentile(latency_ms, 95) FROM api_metrics
WHERE endpoint = '/api/v1/interview/answer'
AND timestamp > now() - 1h
```

**Uptime:**
- Target: 99.5% during demo period
- Monitoring: Sentry + Pingdom

**Report Generation Time:**
- Target: <30s for full PDF
- Breakdown: Score aggregation (5s) + LLM synthesis (15s) + PDF render (10s)

---

## 14. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU cost overrun | Medium | High | Use spot instances, auto-shutdown after 1hr idle, budget alerts |
| STT/TTS latency >5s | Medium | High | Pre-cache common TTS, use faster-whisper INT8, WebSocket streaming |
| LLM evaluation hallucination | High | Critical | Validation rules, rubric grounding, human review of flagged cases |
| Model bias in scoring | Medium | Critical | Diverse test dataset, fairness metrics, human oversight |
| Audio quality issues | High | Medium | Noise reduction pre-processing, confidence thresholds, "repeat" option |
| Timeline slip | High | Medium | Parallel dev tracks, de-scope voice for v1 if needed |

---

## 15. Documentation Deliverables

### 15.1 Technical Documentation
1. **API Reference** - Complete OpenAPI spec with examples
2. **Architecture Diagram** - Visual system overview
3. **Model Cards** - Details on Llama/Whisper/XTTS configuration
4. **Database Schema** - MongoDB collection structures
5. **Deployment Guide** - Step-by-step cloud setup

### 15.2 Code Documentation
1. **README.md** - Project overview, quick start
2. **requirements.txt** - Pinned dependencies
3. **Inline comments** - All complex logic documented
4. **Type hints** - Full Python type coverage

### 15.3 Reports
1. **Final Project Report** - 15-20 pages covering methodology, results, evaluation
2. **Model Evaluation Report** - Benchmarking data, inter-rater reliability
3. **User Guide** - For Unity client team integration

---

## Appendix A: Alternative Models Considered

**STT Alternatives:**
- AssemblyAI API: Higher accuracy but violates IP clause
- Google Speech-to-Text: Similar issue
- Wav2Vec2: Lower accuracy (12-15% WER), but fully local

**TTS Alternatives:**
- Piper TTS: Faster (200ms) but robotic prosody
- Bark: Better emotion control but slower (4s latency)
- ElevenLabs API: Best quality but monthly cost + IP issues

**LLM Alternatives:**
- GPT-4: Better reasoning but API cost ₹15,000+/month
- Mistral-7B: Similar to Llama, slightly lower on instruction tasks
- Qwen-7B: Strong multilingual but English performance below Llama

---

## Appendix B: Sample Interview Flow

**Stage 1: Screening (5 min)**
1. "Tell me about yourself and why this role interests you"
2. "What do you know about our company?"
3. "Walk me through your most recent project"

**Stage 2: Technical (20 min)**
1. "Explain the difference between REST and GraphQL"
2. "How would you optimize a slow database query?"
3. "Design a URL shortener like bit.ly"
4. [Adaptive] If scoring high → System design question
5. [Adaptive] If scoring low → Easier conceptual question

**Stage 3: Behavioral (10 min)**
1. "Tell me about a time you disagreed with your manager"
2. "Describe a project that failed and what you learned"
3. "How do you handle tight deadlines?"

**Stage 4: Wrap-up (2 min)**
1. "Do you have any questions for me?"
2. "Is there anything else you'd like us to know?"

---

## Appendix C: Compliance Checklist

- [x] No external API dependencies (IP ownership)
- [x] Audio encryption at rest (GDPR)
- [x] PII anonymization in logs
- [x] 30-day audio retention policy
- [x] Secure JWT authentication
- [x] Rate limiting (prevent abuse)
- [x] Error logging (no sensitive data)
- [x] CORS configuration (Unity origin only)

---

**END OF DOCUMENT**

*For questions or clarifications, contact: technical-team@hackgenix.com*

### 4.1 Interview Agent Architecture

**Agent Type:** ReAct (Reasoning + Acting) pattern
- **Reasoning:** LLM decides next question based on candidate performance
- **Acting:** Executes question, evaluates answer, updates state

**Core Agent Loop:**
```python
class InterviewAgent:
    def __init__(self, resume: dict, job_desc: dict):
        self.resume = resume
        self.jd = job_desc
        self.state = "initialization"
        self.question_bank = []
        self.transcript = []
        self.scores = {"technical": [], "behavioral": []}
        
    async def initialize(self):
        """Generate personalized interview plan"""
        # 1. Semantic matching
        match_analysis = self.analyze_fit(self.resume, self.jd)
        
        # 2. Generate question bank
        self.question_bank = await self.generate_questions(match_analysis)
        
        # 3. Set difficulty adaptation thresholds
        self.difficulty_threshold = self._calculate_baseline_difficulty()
        
    async def conduct_interview(self):
        """Main interview loop"""
        stages = ["screening", "technical", "behavioral", "wrapup"]
        
        for stage in stages:
            self.state = stage
            questions = self._get_questions_for_stage(stage)
            
            for question in questions:
                # Ask question
                yield {"type": "question", "content": question.text}
                
                # Wait for candidate response (from Unity)
                response = await self.wait_for_response()
                
                # Evaluate answer
                score = await self.evaluate_answer(question, response)
                self.scores[question.category].append(score)
                
                # Adaptive follow-up
                if score.value < 3 and question.allow_followup:
                    followup = await self.generate_followup(question, response, score)
                    yield {"type": "followup", "content": followup}
                
                # Update transcript
                self.transcript.append({
                    "question": question.text,
                    "answer": response.text,
                    "score": score.dict(),
                    "timestamp": datetime.utcnow()
                })
                
                # Early termination check (screening stage)
                if stage == "screening" and self._should_terminate_early():
                    yield {"type": "termination", "reason": "insufficient_qualifications"}
                    return
        
        # Generate report
        report = await self.generate_report()
        yield {"type": "report", "content": report}
```

### 4.2 Question Generation Strategy

**Technical Questions:**

```python
async def generate_technical_questions(self, jd: dict, resume: dict, gaps: list) -> List[Question]:
    """
    Strategy:
    1. Required skills from JD → direct technical questions
    2. Gap areas → probing questions
    3. Experience level → difficulty calibration
    """
    
    prompt = f"""You are an expert technical interviewer. Generate 8 technical interview questions.

JOB REQUIREMENTS:
{jd['required_skills']}
{jd['responsibilities']}

CANDIDATE BACKGROUND:
{resume['skills']}
{resume['experience_years']} years experience

IDENTIFIED GAPS:
{gaps}

INSTRUCTIONS:
1. Generate 3 questions on core required skills: {jd['core_tech_stack']}
2. Generate 2 questions probing gap areas: {gaps}
3. Generate 2 system design questions (if senior role: {jd['seniority_level']})
4. Generate 1 open-ended problem-solving scenario

For EACH question provide:
- Question text (clear, unambiguous)
- Expected answer key points (5-7 bullets)
- Difficulty (1-5 scale)
- Evaluation rubric (what constitutes 1-star vs 5-star answer)
- Time limit (minutes)

OUTPUT FORMAT: JSON array of question objects"""

    response = await self.llm.generate(prompt, max_tokens=2000)
    questions = json.loads(response)
    
    # Validate structure
    for q in questions:
        assert all(k in q for k in ["question", "rubric", "difficulty"])
    
    return [Question(**q) for q in questions]
```

**Behavioral Questions (STAR Method):**

```python
BEHAVIORAL_TEMPLATES = {
    "teamwork": {
        "question": "Tell me about a time you had to collaborate with a difficult team member. How did you handle it?",
        "rubric": {
            "situation": "Clearly described context and team dynamics",
            "task": "Identified their role and responsibility",
            "action": "Specific steps taken, showing emotional intelligence",
            "result": "Positive outcome, lessons learned"
        }
    },
    "leadership": {
        "question": "Describe a situation where you had to lead a project with tight deadlines.",
        "rubric": { ... }
    },
    # ... 10 more categories
}

async def generate_behavioral_questions(self, jd: dict) -> List[Question]:
    """Select 5 behavioral questions based on role requirements"""
    
    # Extract soft skills from JD
    required_traits = extract_soft_skills(jd['description'])
    # e.g., ["leadership", "communication", "problem-solving"]
    
    selected_questions = []
    for trait in required_traits[:5]:
        template = BEHAVIORAL_TEMPLATES.get(trait)
        
        # Personalize using LLM
        personalized = await self.llm.generate(f"""
        Adapt this behavioral question for a {jd['role']} position:
        {template['question']}
        
        Context: {jd['company_culture']}
        Make it specific to {jd['industry']} industry.
        """)
        
        selected_questions.append(Question(
            text=personalized,
            category="behavioral",
            rubric=template['rubric'],
            subcategory=trait
        ))
    
    return selected_questions
```

### 4.3 Adaptive Difficulty

**Performance-Based Adjustment:**
```python
class DifficultyAdapter:
    def __init__(self):
        self.performance_window = []  # Last 3 scores
        
    def should_increase_difficulty(self) -> bool:
        """Increase if recent avg > 4.0"""
        if len(self.performance_window) >= 3:
            return statistics.mean(self.performance_window) > 4.0
        return False
    
    def should_decrease_difficulty(self) -> bool:
        """Decrease if recent avg < 2.5"""
        if len(self.performance_window) >= 3:
            return statistics.mean(self.performance_window) < 2.5
        return False
    
    def get_next_question_difficulty(self, question_bank: List[Question]) -> Question:
        """Select question matching current performance level"""
        if self.should_increase_difficulty():
            return max(question_bank, key=lambda q: q.difficulty)
        elif self.should_decrease_difficulty():
            return min(question_bank, key=lambda q: q.difficulty)
        else:
            # Maintain current difficulty
            target_diff = statistics.mean([q.difficulty for q in question_bank])
            return min(question_bank, key=lambda q: abs(q.difficulty - target_diff))
```

---

## 5. Assessment Engine

### 5.1 LLM-as-Judge Evaluation

**Technical Answer Scoring:**

```python
async def evaluate_technical_answer(self, question: Question, answer: str) -> Score:
    """
    Uses LLM with structured rubric to score answers
    Prevents hallucination by grounding in rubric
    """
    
    evaluation_prompt = f"""You are an expert technical interviewer evaluating a candidate's answer.

QUESTION:
{question.text}

EXPECTED KEY POINTS (from rubric):
{chr(10).join(f"- {point}" for point in question.rubric['key_points'])}

CANDIDATE'S ANSWER:
{answer}

EVALUATION INSTRUCTIONS:
1. Check which key points the candidate covered (list explicitly)
2. Assess technical accuracy (any errors or misconceptions?)
3. Evaluate depth of understanding (surface vs deep knowledge)
4. Consider communication clarity

SCORING SCALE:
5 - Exceptional: All key points covered + additional insights, technically flawless
4 - Strong: Most key points covered, minor gaps, good understanding
3 - Acceptable: Core points covered, some gaps, basic understanding
2 - Weak: Missing several key points, conceptual errors
1 - Insufficient: Mostly incorrect or irrelevant

OUTPUT FORMAT (JSON):
{{
  "score": <1-5>,
  "covered_points": [list of key points mentioned],
  "missed_points": [list of key points not mentioned],
  "technical_errors": [list any factual mistakes],
  "strengths": [2-3 bullet points],
  "weaknesses": [2-3 bullet points],
  "justification": "1-2 sentence reasoning for score"
}}

CRITICAL: Base score ONLY on rubric. Do not invent criteria."""

    response = await self.llm.generate(evaluation_prompt, max_tokens=500)
    evaluation = json.loads(clean_json_response(response))
    
    # Validation: ensure score is justified by covered_points ratio
    coverage_ratio = len(evaluation['covered_points']) / len(question.rubric['key_points'])
    if evaluation['score'] >= 4 and coverage_ratio < 0.7:
        # Override: LLM was too generous
        evaluation['score'] = 3
        evaluation['justification'] += " [Auto-adjusted: insufficient key point coverage]"
    
    return Score(**evaluation)
```

**Behavioral Answer Scoring (STAR Analysis):**

```python
async def evaluate_behavioral_answer(self, question: Question, answer: str) -> Score:
    """
    Specifically checks for STAR method completeness
    """
    
    star_check_prompt = f"""Evaluate this behavioral interview answer using the STAR method.

QUESTION:
{question.text}

CANDIDATE'S ANSWER:
{answer}

ANALYSIS REQUIRED:
1. SITUATION: Did they describe the context clearly? (Yes/No + quote)
2. TASK: Did they explain their responsibility? (Yes/No + quote)
3. ACTION: Did they detail specific steps taken? (Yes/No + quote)
4. RESULT: Did they share outcomes and learnings? (Yes/No + quote)

Additional factors:
- Self-awareness (do they reflect on their actions?)
- Specificity (concrete details vs vague generalities)
- Relevance to question
- Red flags (blaming others, lack of ownership)

SCORING:
5 - Complete STAR, high self-awareness, strong relevance
4 - Complete STAR, good detail, minor gaps in reflection
3 - Partial STAR (missing 1 element), adequate relevance
2 - Incomplete STAR (missing 2+ elements), vague
1 - No STAR structure, irrelevant or concerning response

OUTPUT JSON:
{{
  "score": <1-5>,
  "star_completeness": {{
    "situation": {{" present": true/false, "quote": "..." }},
    "task": {{"present": true/false, "quote": "..."}},
    "action": {{"present": true/false, "quote": "..."}},
    "result": {{"present": true/false, "quote": "..."}}
  }},
  "self_awareness_level": "high|medium|low",
  "red_flags": [list any concerning patterns],
  "justification": "..."
}}"""

    response = await self.llm.generate(star_check_prompt, max_tokens=600)
    return Score(**json.loads(clean_json_response(response)))
```

### 5.2 Hallucination Prevention in Evaluation

**Problem:** LLM-as-judge might give scores not justified by rubric.

**Solution:** Post-evaluation validation

```python
def validate_evaluation(self, evaluation: dict, question: Question) -> dict:
    """
    Checks if LLM evaluation is grounded in rubric
    """
    
    # Rule 1: Score must correlate with covered_points ratio
    coverage = len(evaluation['covered_points']) / len(question.rubric['key_points'])
    expected_score_range = {
        (0.0, 0.3): (1, 2),
        (0.3, 0.6): (2, 3),
        (0.6, 0.8): (3, 4),
        (0.8, 1.0): (4, 5)
    }
    
    for (low, high), (min_score, max_score) in expected_score_range.items():
        if low <= coverage < high:
            if not (min_score <= evaluation['score'] <= max_score):
                # Override score
                evaluation['score'] = max_score if coverage > 0.7 else min_score
                evaluation['validation_override'] = True
                break
    
    # Rule 2: Technical errors should cap score at 3
    if evaluation.get('technical_errors') and len(evaluation['technical_errors']) > 0:
        if evaluation['score'] > 3:
            evaluation['score'] = 3
            evaluation['validation_override'] = True
    
    # Rule 3: Red flags in behavioral → cap at 2
    if evaluation.get('red_flags') and len(evaluation['red_flags']) > 0:
        if evaluation['score'] > 2:
            evaluation['score'] = 2
            evaluation['validation_override'] = True
    
    return evaluation
```

### 5.3 Communication Quality Metrics

**Automated Metrics (rule-based):**

```python
from textblob import TextBlob
import language_tool_python

class CommunicationAnalyzer:
    def __init__(self):
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
    
    def analyze(self, answer: str) -> dict:
        return {
            "word_count": len(answer.split()),
            "avg_sentence_length": self._avg_sentence_length(answer),
            "grammar_errors": len(self.grammar_tool.check(answer)),
            "filler_words": self._count_fillers(answer),  # um, uh, like
            "clarity_score": self._calculate_clarity(answer),  # Flesch reading ease
            "confidence_indicators": self._detect_confidence(answer)  # "I think", "maybe"
        }
    
    def _count_fillers(self, text: str) -> int:
        fillers = ["um", "uh", "like", "you know", "sort of", "kind of"]
        return sum(text.lower().count(f" {filler} ") for filler in fillers)
    
    def _calculate_clarity(self, text: str) -> float:
        """Flesch Reading Ease score (0-100, higher = clearer)"""
        blob = TextBlob(text)
        # Simplified calculation
        words = len(blob.words)
        sentences = len(blob.sentences)