#!/usr/bin/env python
"""
Voice Integration E2E Test - Full Voice Interview Flow.

This script tests the complete voice interview pipeline:
1. TTS: Generate spoken interview questions
2. STT: Transcribe candidate responses
3. Full loop: Question -> Audio -> Answer -> Transcription -> Evaluation

Run with: python scripts/test_voice_integration.py

Requires: Whisper model will be downloaded on first run (~150MB for base)
"""
import asyncio
import sys
import os
import tempfile
import wave
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ANSI colors for output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.RESET}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[PASS] {text}{Colors.RESET}")


def print_fail(text: str):
    print(f"{Colors.RED}[FAIL] {text}{Colors.RESET}")


def print_info(text: str):
    print(f"{Colors.CYAN}[INFO] {text}{Colors.RESET}")


def print_warn(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.RESET}")


# Sample interview questions for testing
SAMPLE_QUESTIONS = [
    "Tell me about a challenging technical project you worked on recently.",
    "How would you design a rate limiting API for a high-traffic service?",
    "Describe a time when you had to resolve a conflict with a team member.",
]


async def test_tts_provider():
    """Test the Text-to-Speech provider."""
    print_section("Testing TTS Provider")
    
    try:
        from src.providers.tts import get_tts_provider_async
        
        tts = await get_tts_provider_async(rate=150, volume=1.0)
        
        # Get available voices
        voices = tts.get_voices()
        print_success(f"TTS Provider initialized: {len(voices)} voices available")
        
        if voices:
            print_info(f"Sample voice: {voices[0].name}")
        
        # Synthesize a test question
        test_text = "Hello, welcome to your technical interview."
        result = await tts.synthesize(test_text)
        
        print_success(f"Synthesized audio: {result.duration_seconds:.2f}s, {len(result.audio_data)} bytes")
        
        # Verify it's valid WAV data
        assert result.audio_data[:4] == b'RIFF', "Invalid WAV header"
        assert result.sample_rate > 0, "Invalid sample rate"
        
        print_success("TTS audio validation passed")
        
        return True, result.audio_data
        
    except Exception as e:
        print_fail(f"TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_stt_provider(audio_data: bytes = None):
    """Test the Speech-to-Text provider."""
    print_section("Testing STT Provider (Faster-Whisper)")
    
    try:
        from src.providers.stt import get_faster_whisper_provider_async
        
        print_info("Loading Faster-Whisper model (this may take a moment on first run)...")
        stt = await get_faster_whisper_provider_async(model_name="base")
        
        model_info = stt.get_model_info()
        print_success(f"STT Provider initialized: {model_info['model_name']} on {model_info['device']}")
        
        # If we have TTS audio, try to transcribe it
        if audio_data:
            print_info("Transcribing TTS-generated audio...")
            
            result = await stt.transcribe(audio_data)
            
            print_success(f"Transcription: '{result.text}'")
            print_info(f"Language: {result.language}, Confidence: {result.confidence:.2f}")
            print_info(f"Duration: {result.duration_seconds:.2f}s")
            
            # Basic validation
            assert len(result.text) > 0, "Empty transcription"
            
            print_success("STT transcription validation passed")
        else:
            print_warn("No audio data provided, skipping transcription test")
        
        return True
        
    except Exception as e:
        print_fail(f"STT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_interview_question_synthesis():
    """Test synthesizing interview questions."""
    print_section("Testing Interview Question Synthesis")
    
    try:
        from src.providers.tts import get_tts_provider_async
        
        tts = await get_tts_provider_async()
        
        for i, question in enumerate(SAMPLE_QUESTIONS, 1):
            result = await tts.synthesize_interview_question(
                question=question,
                question_number=i,
                total_questions=len(SAMPLE_QUESTIONS),
            )
            
            print_success(f"Q{i}: {result.duration_seconds:.1f}s audio ({len(result.audio_data)} bytes)")
        
        print_success("All interview questions synthesized successfully")
        return True
        
    except Exception as e:
        print_fail(f"Interview question synthesis failed: {e}")
        return False


async def test_transcribe_with_context():
    """Test transcription with interview context."""
    print_section("Testing Context-Aware Transcription")
    
    try:
        from src.providers.stt import get_faster_whisper_provider_async
        from src.providers.tts import get_tts_provider_async
        
        # Generate audio with technical content
        tts = await get_tts_provider_async()
        
        technical_answer = (
            "I would use Python with FastAPI for the backend, "
            "PostgreSQL for the database, and Redis for caching. "
            "For deployment, I'd containerize with Docker and orchestrate with Kubernetes."
        )
        
        audio_result = await tts.synthesize(technical_answer)
        print_info(f"Generated technical answer audio: {audio_result.duration_seconds:.1f}s")
        
        # Transcribe with interview context
        stt = await get_faster_whisper_provider_async(model_name="base")
        
        result = await stt.transcribe_with_interview_context(
            audio_data=audio_result.audio_data,
            context="Backend engineering interview",
            technical_terms=["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        )
        
        print_success(f"Context-aware transcription: '{result.text[:80]}...'")
        print_info(f"Segments: {len(result.segments)}")
        
        # Check if technical terms are recognized
        text_lower = result.text.lower()
        recognized_terms = []
        for term in ["python", "fastapi", "postgresql", "redis", "docker", "kubernetes"]:
            if term in text_lower:
                recognized_terms.append(term)
        
        print_info(f"Recognized technical terms: {', '.join(recognized_terms)}")
        
        return True
        
    except Exception as e:
        print_fail(f"Context-aware transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_voice_loop():
    """Test a complete voice interview loop."""
    print_section("Testing Full Voice Interview Loop")
    
    try:
        from src.providers.tts import get_tts_provider_async
        from src.providers.stt import get_faster_whisper_provider_async
        from src.services.answer_evaluator import AnswerEvaluator
        from src.services.prompts import InterviewStage
        from src.providers.llm import get_llm_provider
        
        tts = await get_tts_provider_async()
        stt = await get_faster_whisper_provider_async(model_name="base")
        
        # Step 1: Generate question audio
        question = "How would you handle a database query that's running too slowly?"
        print_info(f"Question: {question}")
        
        question_audio = await tts.synthesize_interview_question(question, 1, 1)
        print_success(f"Question audio: {question_audio.duration_seconds:.1f}s")
        
        # Step 2: Simulate candidate answer (in real system, this would be recorded audio)
        simulated_answer = (
            "First, I would analyze the query using EXPLAIN ANALYZE to understand the execution plan. "
            "Then I'd look for missing indexes, particularly on columns used in WHERE clauses and JOINs. "
            "If the table is large, I might consider partitioning or adding read replicas. "
            "I'd also check if the query can be rewritten to be more efficient."
        )
        
        answer_audio = await tts.synthesize(simulated_answer)
        print_info(f"Simulated answer audio: {answer_audio.duration_seconds:.1f}s")
        
        # Step 3: Transcribe the answer
        transcription = await stt.transcribe_with_interview_context(
            answer_audio.audio_data,
            context="Database optimization interview question",
            technical_terms=["EXPLAIN", "ANALYZE", "index", "partition", "replica"],
        )
        print_success(f"Transcribed: '{transcription.text[:80]}...'")
        
        # Step 4: Evaluate the transcribed answer
        try:
            llm = await get_llm_provider()
            evaluator = AnswerEvaluator(llm_provider=llm)
            
            evaluation = await evaluator.evaluate_answer(
                question=question,
                answer=transcription.text,
                expected_points=["query analysis", "indexing", "optimization"],
                stage=InterviewStage.TECHNICAL,
                validate=False,
            )
            
            print_success(f"Evaluation score: {evaluation.scores.overall:.1f}/100")
            print_info(f"Recommendation: {evaluation.recommendation.value}")
            
        except Exception as e:
            print_warn(f"LLM evaluation skipped (Ollama may not be running): {e}")
        
        print_success("Full voice interview loop completed!")
        return True
        
    except Exception as e:
        print_fail(f"Full voice loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_voice_tests():
    """Run all voice integration tests."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("="*70)
    print("       VOICE INTEGRATION E2E TEST")
    print("       STT (Whisper) + TTS (pyttsx3)")
    print("="*70)
    print(Colors.RESET)
    
    start_time = datetime.now()
    results = {}
    
    # Test 1: TTS Provider
    print_header("PHASE 1: Text-to-Speech Provider")
    tts_success, audio_data = await test_tts_provider()
    results["tts_provider"] = tts_success
    
    # Test 2: STT Provider
    print_header("PHASE 2: Speech-to-Text Provider")
    stt_success = await test_stt_provider(audio_data)
    results["stt_provider"] = stt_success
    
    # Test 3: Interview Question Synthesis
    print_header("PHASE 3: Interview Question Synthesis")
    synthesis_success = await test_interview_question_synthesis()
    results["question_synthesis"] = synthesis_success
    
    # Test 4: Context-Aware Transcription
    print_header("PHASE 4: Context-Aware Transcription")
    context_success = await test_transcribe_with_context()
    results["context_transcription"] = context_success
    
    # Test 5: Full Voice Loop
    print_header("PHASE 5: Full Voice Interview Loop")
    loop_success = await test_full_voice_loop()
    results["voice_loop"] = loop_success
    
    # Summary
    print_header("TEST RESULTS SUMMARY")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")
    print(f"Duration: {duration:.1f} seconds")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}")
        print("="*70)
        print("              ALL VOICE TESTS PASSED!")
        print("        Voice integration is fully operational.")
        print("="*70)
        print(Colors.RESET)
    else:
        print(f"\n{Colors.YELLOW}Some tests failed. Check output above for details.{Colors.RESET}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_voice_tests())
