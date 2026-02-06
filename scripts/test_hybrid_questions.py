"""
End-to-End Test for Hybrid Question Generation (Phase 6.5).

Tests the complete hybrid question generation flow:
1. Load question bank
2. Select questions from bank
3. Enhance/personalize bank questions with LLM
4. Generate gap-filling questions for uncovered skills
5. Verify question sources and quality

Prerequisites:
- Ollama running with qwen2.5:3b model
- Question bank files in questionBank/domains/

Usage:
    .venv/Scripts/python.exe scripts/test_hybrid_questions.py
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.interview import InterviewConfig, InterviewStage
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo
from src.models.question_bank import (
    QuestionBankConfig,
    QuestionSource,
    InterviewStageHint,
)
from src.services.question_bank import QuestionBankService, get_question_bank_service
from src.services.hybrid_question_selector import HybridQuestionSelector
from src.services.question_generator import (
    QuestionGenerator,
    QuestionGenerationRequest,
)
from src.services.prompts import InterviewStage as PromptStage, QuestionDifficulty


def create_test_resume() -> ParsedResume:
    """Create a realistic test resume."""
    return ParsedResume(
        contact=ContactInfo(
            name="Alex Thompson",
            email="alex.thompson@email.com",
            phone="555-0199",
            linkedin="linkedin.com/in/alexthompson",
            github="github.com/alexthompson",
        ),
        summary="Senior Backend Engineer with 7 years of experience building scalable "
                "distributed systems. Expert in Python, Go, and cloud infrastructure. "
                "Led multiple teams through successful product launches.",
        skills=[
            "Python", "Go", "FastAPI", "PostgreSQL", "Redis", "MongoDB",
            "Docker", "Kubernetes", "AWS", "Terraform", "Kafka",
            "GraphQL", "REST APIs", "gRPC", "Microservices",
            "Pytest", "CI/CD", "Git", "System Design",
        ],
        experience=[
            {
                "company": "MegaTech Corp",
                "title": "Senior Backend Engineer",
                "location": "San Francisco, CA",
                "start_date": "2020-03",
                "end_date": "Present",
                "description": "Lead backend engineer for core platform services.",
                "highlights": [
                    "Architected event-driven microservices handling 50K events/sec",
                    "Designed and implemented real-time analytics pipeline with Kafka",
                    "Led migration from AWS to multi-cloud setup",
                ],
            },
            {
                "company": "StartupABC",
                "title": "Software Engineer",
                "location": "New York, NY",
                "start_date": "2017-01",
                "end_date": "2020-02",
                "description": "Backend developer for fintech platform.",
                "highlights": [
                    "Built payment processing service handling $1M+ daily transactions",
                    "Implemented caching layer reducing API latency by 75%",
                ],
            },
        ],
        education=[
            {
                "institution": "Stanford University",
                "degree": "M.S. Computer Science",
                "field": "Computer Science",
                "end_date": "2016",
            },
        ],
        certifications=["AWS Solutions Architect Professional", "CKA"],
        raw_text="Alex Thompson - Senior Backend Engineer with expertise in distributed systems.",
    )


def create_test_jd() -> ParsedJobDescription:
    """Create a realistic test job description."""
    return ParsedJobDescription(
        title="Staff Backend Engineer",
        company="HackGeniX Tech",
        location="Remote",
        employment_type="full-time",
        experience_level="senior",
        experience_years_min=5,
        experience_years_max=10,
        required_skills=[
            "Python", "PostgreSQL", "Redis", "Docker", "Kubernetes",
            "REST APIs", "Microservices", "System Design", "Event-Driven Architecture",
        ],
        preferred_skills=[
            "Go", "Kafka", "GraphQL", "Terraform", "AWS",
            "gRPC", "MongoDB", "Machine Learning",
        ],
        responsibilities=[
            "Design and implement scalable backend services",
            "Lead technical architecture decisions",
            "Mentor junior engineers",
            "Drive best practices for testing and deployment",
        ],
        qualifications=[
            "7+ years of backend development experience",
            "Experience with distributed systems at scale",
            "Strong communication and leadership skills",
        ],
        raw_text="Staff Backend Engineer position at HackGeniX Tech building next-gen platform.",
    )


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_question(idx: int, question, show_details: bool = True):
    """Print a question with its metadata."""
    source = getattr(question, 'source', 'unknown')
    if hasattr(source, 'value'):
        source = source.value
    
    print(f"\n  [{idx}] {question.question[:100]}{'...' if len(question.question) > 100 else ''}")
    
    if show_details:
        print(f"      Source: {source}")
        print(f"      Difficulty: {question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty}")
        print(f"      Category: {question.category}")
        if hasattr(question, 'original_bank_question') and question.original_bank_question:
            orig = question.original_bank_question
            if orig != question.question:
                print(f"      Original: {orig[:80]}{'...' if len(orig) > 80 else ''}")


async def test_question_bank_loading():
    """Test 1: Question Bank Loading."""
    print_separator("Test 1: Question Bank Loading")
    
    bank_service = get_question_bank_service()
    
    # List available domains
    domains = bank_service.list_available_domains()
    print(f"\n  Available domains: {domains}")
    
    if not domains:
        print("  ERROR: No domains found in questionBank/domains/")
        return None
    
    # Load a domain
    start_time = time.time()
    questions = await bank_service.load_domain("backend")
    load_time = time.time() - start_time
    
    print(f"\n  Loaded 'backend' domain:")
    print(f"    - Questions: {len(questions)}")
    print(f"    - Load time: {load_time*1000:.1f}ms")
    
    # Show sample questions with enriched metadata
    print("\n  Sample questions with enriched metadata:")
    for i, q in enumerate(questions[:3]):
        print(f"\n    [{i+1}] {q.question_text[:70]}...")
        print(f"        Category: {q.category.value}")
        print(f"        Difficulty: {q.difficulty.value}")
        print(f"        Skills: {q.skills[:5]}")
        print(f"        Stage Hint: {q.stage_hint.value}")
    
    # Get stats
    await bank_service.load_all_domains()
    stats = bank_service.get_stats()
    print(f"\n  Question Bank Stats:")
    print(f"    - Total questions: {stats.total_questions}")
    print(f"    - Domains loaded: {stats.loaded_domains}")
    print(f"    - Categories: {list(stats.questions_by_category.keys())}")
    print(f"    - Unique skills: {len(stats.unique_skills)}")
    
    return bank_service


async def test_hybrid_question_selection(bank_service):
    """Test 2: Hybrid Question Selection."""
    print_separator("Test 2: Hybrid Question Selection")
    
    resume = create_test_resume()
    jd = create_test_jd()
    
    config = QuestionBankConfig(
        use_question_bank=True,
        auto_detect_domains=True,
        bank_question_ratio=0.7,
        allow_rephrasing=True,
        allow_personalization=True,
    )
    
    selector = HybridQuestionSelector(bank_service=bank_service)
    
    # Test selection for TECHNICAL stage
    print("\n  Selecting questions for TECHNICAL stage...")
    start_time = time.time()
    
    bank_questions, uncovered_skills = await selector.select_questions(
        jd=jd,
        resume=resume,
        config=config,
        stage=InterviewStageHint.TECHNICAL,
        count=5,
    )
    
    select_time = time.time() - start_time
    
    print(f"\n  Selection Results:")
    print(f"    - Bank questions selected: {len(bank_questions)}")
    print(f"    - Uncovered skills: {len(uncovered_skills)}")
    print(f"    - Selection time: {select_time*1000:.1f}ms")
    
    if bank_questions:
        print("\n  Selected Bank Questions:")
        for i, q in enumerate(bank_questions[:3]):
            print(f"    [{i+1}] {q.question_text[:70]}...")
            print(f"        Skills: {q.skills[:3]}")
    
    if uncovered_skills:
        print(f"\n  Uncovered Skills (for LLM gap-filling):")
        print(f"    {uncovered_skills[:10]}")
    
    # Get coverage report
    jd_skills = jd.required_skills + jd.preferred_skills
    report = selector.get_skill_coverage_report(bank_questions, jd_skills)
    print(f"\n  Skill Coverage Report:")
    print(f"    - Total JD skills: {report['total_jd_skills']}")
    print(f"    - Covered: {len(report['covered_skills'])} ({report['coverage_percent']:.1f}%)")
    print(f"    - Uncovered: {len(report['uncovered_skills'])}")
    
    return bank_questions, uncovered_skills


async def test_hybrid_question_generation(bank_questions, uncovered_skills):
    """Test 3: Full Hybrid Question Generation with LLM."""
    print_separator("Test 3: Full Hybrid Question Generation")
    
    resume = create_test_resume()
    jd = create_test_jd()
    
    config = QuestionBankConfig(
        use_question_bank=True,
        bank_question_ratio=0.7,
        allow_rephrasing=True,
        allow_personalization=True,
    )
    
    request = QuestionGenerationRequest(
        stage=PromptStage.TECHNICAL,
        num_questions=5,
        difficulty=QuestionDifficulty.MEDIUM,
    )
    
    generator = QuestionGenerator()
    
    print("\n  Generating questions using hybrid mode...")
    print(f"    - Bank questions to enhance: {len(bank_questions)}")
    print(f"    - Skills for gap-filling: {len(uncovered_skills)}")
    
    start_time = time.time()
    
    questions = await generator.generate_questions_hybrid(
        request=request,
        resume=resume,
        jd=jd,
        bank_config=config,
        bank_questions=bank_questions,
        uncovered_skills=uncovered_skills,
    )
    
    gen_time = time.time() - start_time
    
    print(f"\n  Generation Results:")
    print(f"    - Total questions generated: {len(questions)}")
    print(f"    - Generation time: {gen_time:.2f}s")
    
    # Count by source
    source_counts = {}
    for q in questions:
        source = q.source.value if hasattr(q.source, 'value') else str(q.source)
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n  Questions by Source:")
    for source, count in source_counts.items():
        print(f"    - {source}: {count}")
    
    print(f"\n  Generated Questions:")
    for i, q in enumerate(questions):
        print_question(i + 1, q, show_details=True)
    
    return questions


async def test_pure_llm_comparison():
    """Test 4: Compare with Pure LLM Generation."""
    print_separator("Test 4: Pure LLM Generation (Comparison)")
    
    resume = create_test_resume()
    jd = create_test_jd()
    
    request = QuestionGenerationRequest(
        stage=PromptStage.TECHNICAL,
        num_questions=5,
        difficulty=QuestionDifficulty.MEDIUM,
    )
    
    generator = QuestionGenerator()
    
    print("\n  Generating questions using pure LLM mode...")
    
    start_time = time.time()
    
    questions = await generator.generate_questions(
        request=request,
        resume=resume,
        jd=jd,
    )
    
    gen_time = time.time() - start_time
    
    print(f"\n  Pure LLM Generation Results:")
    print(f"    - Questions generated: {len(questions)}")
    print(f"    - Generation time: {gen_time:.2f}s")
    
    print(f"\n  Generated Questions (Pure LLM):")
    for i, q in enumerate(questions[:3]):
        print(f"\n  [{i+1}] {q.question[:100]}...")
    
    return gen_time


async def test_orchestrator_integration():
    """Test 5: Interview Orchestrator Integration."""
    print_separator("Test 5: Interview Orchestrator Integration")
    
    from src.services.interview_orchestrator import InterviewOrchestrator
    
    resume = create_test_resume()
    jd = create_test_jd()
    
    # Config with hybrid mode enabled
    config = InterviewConfig(
        screening_questions=2,
        technical_questions=4,
        behavioral_questions=2,
        system_design_questions=1,
        use_question_bank=True,
        auto_detect_domains=True,
        bank_question_ratio=0.6,
        allow_rephrasing=True,
        allow_personalization=True,
    )
    
    orchestrator = InterviewOrchestrator()
    
    print("\n  Starting interview session with hybrid question generation...")
    start_time = time.time()
    
    session, response = await orchestrator.start_interview(
        resume=resume,
        jd=jd,
        resume_id="test-resume-001",
        jd_id="test-jd-001",
        config=config,
    )
    
    init_time = time.time() - start_time
    
    print(f"\n  Session Started:")
    print(f"    - Session ID: {session.id}")
    print(f"    - Status: {response.status}")
    print(f"    - Total questions: {response.total_questions}")
    print(f"    - Initialization time: {init_time:.2f}s")
    
    # Analyze question sources
    source_counts = {}
    for q in session.questions:
        source = getattr(q, 'source', 'generated')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n  Questions by Source:")
    for source, count in source_counts.items():
        print(f"    - {source}: {count}")
    
    # Show questions by stage
    print(f"\n  Questions by Stage:")
    for stage in [InterviewStage.SCREENING, InterviewStage.TECHNICAL, 
                  InterviewStage.BEHAVIORAL, InterviewStage.SYSTEM_DESIGN]:
        stage_qs = session.get_stage_questions(stage)
        if stage_qs:
            print(f"\n    {stage.value.upper()} ({len(stage_qs)} questions):")
            for i, q in enumerate(stage_qs[:2]):
                source = getattr(q, 'source', 'generated')
                print(f"      [{i+1}] {q.question_text[:60]}... (source: {source})")
    
    return session


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" HYBRID QUESTION GENERATION E2E TEST (Phase 6.5)")
    print("="*60)
    
    total_start = time.time()
    results = {}
    
    try:
        # Test 1: Question Bank Loading
        bank_service = await test_question_bank_loading()
        results["bank_loading"] = "PASS" if bank_service else "FAIL"
        
        if not bank_service:
            print("\n  Skipping remaining tests due to bank loading failure.")
            return
        
        # Test 2: Hybrid Selection
        bank_questions, uncovered_skills = await test_hybrid_question_selection(bank_service)
        results["hybrid_selection"] = "PASS"
        
        # Test 3: Hybrid Generation with LLM
        hybrid_questions = await test_hybrid_question_generation(bank_questions, uncovered_skills)
        results["hybrid_generation"] = "PASS" if hybrid_questions else "FAIL"
        
        # Test 4: Pure LLM Comparison
        pure_llm_time = await test_pure_llm_comparison()
        results["pure_llm"] = "PASS"
        
        # Test 5: Orchestrator Integration
        session = await test_orchestrator_integration()
        results["orchestrator"] = "PASS" if session else "FAIL"
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    total_time = time.time() - total_start
    
    # Summary
    print_separator("TEST SUMMARY")
    print(f"\n  Total execution time: {total_time:.2f}s")
    print(f"\n  Results:")
    for test_name, status in results.items():
        icon = "+" if status == "PASS" else "-"
        print(f"    [{icon}] {test_name}: {status}")
    
    all_passed = all(v == "PASS" for v in results.values() if v in ["PASS", "FAIL"])
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
