#!/usr/bin/env python
"""
End-to-End LLM Integration Test Script.

This script tests the complete LLM interview flow:
1. LLM Provider connection (vLLM or Ollama fallback)
2. Question generation for different interview stages
3. Answer evaluation with LLM-as-judge
4. Hallucination detection
5. Interview report generation

Run with: python scripts/test_llm_integration.py

Note: Requires either vLLM (http://localhost:8001) or Ollama (http://localhost:11434) running.
"""
import asyncio
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.providers.llm import get_llm_provider, LLMProviderFactory
from src.providers.llm.base import Message, GenerationConfig, system_message, user_message
from src.services.question_generator import QuestionGenerator, QuestionGenerationRequest
from src.services.answer_evaluator import AnswerEvaluator, InterviewStage
from src.services.prompts import InterviewStage, QuestionDifficulty
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo, Experience


# ANSI colors for output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


def create_test_resume() -> ParsedResume:
    """Create a realistic test resume."""
    return ParsedResume(
        contact=ContactInfo(
            name="Alex Johnson",
            email="alex.johnson@example.com",
            phone="+1-555-123-4567",
        ),
        summary="Senior Software Engineer with 6 years of experience in backend development, "
                "specializing in Python, microservices, and cloud architecture. "
                "Passionate about building scalable systems and mentoring junior developers.",
        skills=[
            "Python", "Django", "FastAPI", "PostgreSQL", "Redis",
            "AWS", "Docker", "Kubernetes", "GraphQL", "REST APIs",
            "CI/CD", "Git", "Agile", "System Design", "Machine Learning"
        ],
        experience=[
            Experience(
                company="TechCorp Inc.",
                title="Senior Software Engineer",
                start_date="2021-01",
                end_date="present",
                description="Lead backend development for a high-traffic e-commerce platform",
                highlights=[
                    "Designed and implemented microservices architecture handling 10M+ daily requests",
                    "Reduced API latency by 40% through caching and optimization",
                    "Mentored team of 4 junior developers",
                ],
            ),
            Experience(
                company="StartupXYZ",
                title="Software Engineer",
                start_date="2018-06",
                end_date="2020-12",
                description="Full-stack development for a SaaS analytics platform",
                highlights=[
                    "Built real-time data pipeline using Python and Kafka",
                    "Implemented GraphQL API for mobile applications",
                ],
            ),
        ],
        raw_text="Alex Johnson - Senior Software Engineer with 6 years experience...",
    )


def create_test_jd() -> ParsedJobDescription:
    """Create a realistic test job description."""
    return ParsedJobDescription(
        title="Senior Backend Engineer",
        company="InnovateTech",
        required_skills=[
            "Python", "Django or FastAPI", "PostgreSQL", "Redis",
            "AWS", "Docker", "REST APIs", "Microservices"
        ],
        preferred_skills=[
            "Kubernetes", "GraphQL", "Machine Learning", "System Design"
        ],
        responsibilities=[
            "Design and implement scalable backend services",
            "Build and maintain RESTful and GraphQL APIs",
            "Optimize database queries and system performance",
            "Mentor junior team members",
            "Participate in architecture decisions",
        ],
        experience_years_min=5,
        experience_years_max=10,
        raw_text="We are looking for a Senior Backend Engineer to join our team...",
    )


async def test_llm_provider_connection():
    """Test LLM provider connection and basic generation."""
    print_header("Test 1: LLM Provider Connection")
    
    try:
        # Test async provider
        provider = await get_llm_provider()
        print_info(f"Provider type: {type(provider).__name__}")
        
        # Test basic generation
        messages = [
            system_message("You are a helpful assistant."),
            user_message("Say 'Hello, Interview System!' in exactly 5 words."),
        ]
        
        config = GenerationConfig(max_tokens=50, temperature=0.1)
        response = await provider.generate(messages, config)
        
        print_success(f"LLM Response: {response.content[:100]}...")
        print_info(f"Tokens used: {response.tokens_used}")
        print_info(f"Model: {response.model}")
        
        return True, provider
        
    except Exception as e:
        print_error(f"LLM connection failed: {e}")
        print_warning("Make sure vLLM or Ollama is running")
        return False, None


async def test_question_generation(provider):
    """Test question generation for different stages."""
    print_header("Test 2: Question Generation")
    
    try:
        generator = QuestionGenerator(llm_provider=provider)
        resume = create_test_resume()
        jd = create_test_jd()
        
        results = {}
        
        # Test screening questions
        print_info("Generating screening questions...")
        screening_qs = await generator.generate_screening_questions(resume, jd, num_questions=2)
        print_success(f"Generated {len(screening_qs)} screening questions")
        for i, q in enumerate(screening_qs, 1):
            print(f"  {i}. {q.question[:80]}...")
        results['screening'] = screening_qs
        
        # Test technical questions
        print_info("\nGenerating technical questions...")
        technical_qs = await generator.generate_technical_questions(
            resume, jd, num_questions=3, focus_areas=["Python", "APIs"]
        )
        print_success(f"Generated {len(technical_qs)} technical questions")
        for i, q in enumerate(technical_qs, 1):
            print(f"  {i}. [{q.difficulty.value}] {q.question[:70]}...")
        results['technical'] = technical_qs
        
        # Test behavioral questions
        print_info("\nGenerating behavioral questions...")
        behavioral_qs = await generator.generate_behavioral_questions(resume, jd, num_questions=2)
        print_success(f"Generated {len(behavioral_qs)} behavioral questions")
        for i, q in enumerate(behavioral_qs, 1):
            print(f"  {i}. {q.question[:80]}...")
        results['behavioral'] = behavioral_qs
        
        return True, results
        
    except Exception as e:
        print_error(f"Question generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


async def test_follow_up_generation(provider, technical_question: str):
    """Test follow-up question generation."""
    print_header("Test 3: Follow-up Question Generation")
    
    try:
        generator = QuestionGenerator(llm_provider=provider)
        
        # Simulate a candidate answer
        candidate_answer = """
        I would design a REST API using FastAPI with proper HTTP methods. 
        GET for retrieving resources, POST for creating, PUT for updating, 
        and DELETE for removing. I'd use Pydantic for validation.
        """
        
        print_info(f"Original question: {technical_question[:60]}...")
        print_info(f"Candidate answer: {candidate_answer[:60]}...")
        
        follow_up = await generator.generate_follow_up(
            original_question=technical_question,
            candidate_answer=candidate_answer,
            evaluation_summary="Good basic understanding but lacks depth on error handling and versioning.",
        )
        
        if follow_up:
            print_success(f"Follow-up question: {follow_up}")
        else:
            print_warning("No follow-up generated (may be intentional)")
        
        return True
        
    except Exception as e:
        print_error(f"Follow-up generation failed: {e}")
        return False


async def test_answer_evaluation(provider):
    """Test answer evaluation with LLM-as-judge."""
    print_header("Test 4: Answer Evaluation (LLM-as-Judge)")
    
    try:
        evaluator = AnswerEvaluator(llm_provider=provider)
        
        # Test technical answer evaluation
        print_info("Evaluating a technical answer...")
        
        tech_eval = await evaluator.evaluate_answer(
            question="Explain how you would design a rate limiting system for an API.",
            answer="""
            I would implement a token bucket algorithm. Each user gets a bucket with a 
            fixed number of tokens that refill at a constant rate. When a request comes in, 
            we check if there are tokens available. If yes, we consume one and allow the request. 
            If not, we return a 429 Too Many Requests response.
            
            For distributed systems, I'd store the token counts in Redis to ensure consistency 
            across multiple API servers. We could also implement different rate limits for 
            different API endpoints or user tiers.
            """,
            expected_points=[
                "Token bucket or leaky bucket algorithm",
                "Distributed storage (Redis/Memcached)",
                "HTTP 429 status code",
                "Per-user or per-IP tracking",
            ],
            stage=InterviewStage.TECHNICAL,
            validate=True,
        )
        
        print_success(f"Technical evaluation completed!")
        print(f"  Overall Score: {tech_eval.scores.overall:.1f}/100")
        print(f"  Technical Accuracy: {tech_eval.scores.technical_accuracy:.1f}")
        print(f"  Completeness: {tech_eval.scores.completeness:.1f}")
        print(f"  Recommendation: {tech_eval.recommendation.value}")
        print(f"  Validated: {tech_eval.is_validated}")
        if tech_eval.strengths:
            print(f"  Strengths: {', '.join(tech_eval.strengths[:2])}")
        
        # Test behavioral answer evaluation
        print_info("\nEvaluating a behavioral answer...")
        
        behavioral_eval = await evaluator.evaluate_behavioral_answer(
            question="Tell me about a time when you had to deal with a difficult team member.",
            answer="""
            At my previous company, I worked with a developer who was very resistant to code reviews. 
            He would get defensive and dismiss feedback.
            
            The situation was affecting team morale and code quality. My task was to find a way to 
            improve collaboration without creating more conflict.
            
            I started by having a one-on-one conversation to understand his perspective. It turned out 
            he felt his experience wasn't being valued. I suggested we rotate who leads code reviews 
            and established clearer guidelines. I also made sure to acknowledge his expertise publicly.
            
            The result was a 50% improvement in code review completion time and he became one of 
            our most constructive reviewers. He even thanked me later for the approach.
            """,
            competency="conflict-resolution",
            red_flags=["Blaming others", "Avoiding responsibility", "Vague details"],
            green_flags=["Taking initiative", "Empathy", "Measurable outcomes", "Self-reflection"],
            validate=True,
        )
        
        print_success(f"Behavioral evaluation completed!")
        print(f"  Overall Score: {behavioral_eval.scores.overall:.1f}/100")
        if behavioral_eval.star_scores:
            print(f"  STAR Scores - S:{behavioral_eval.star_scores.situation:.0f} "
                  f"T:{behavioral_eval.star_scores.task:.0f} "
                  f"A:{behavioral_eval.star_scores.action:.0f} "
                  f"R:{behavioral_eval.star_scores.result:.0f}")
        print(f"  Recommendation: {behavioral_eval.recommendation.value}")
        print(f"  Validated: {behavioral_eval.is_validated}")
        
        return True, [tech_eval, behavioral_eval]
        
    except Exception as e:
        print_error(f"Answer evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


async def test_interview_report(provider, evaluations):
    """Test interview report generation."""
    print_header("Test 5: Interview Report Generation")
    
    try:
        evaluator = AnswerEvaluator(llm_provider=provider)
        
        print_info("Generating comprehensive interview report...")
        
        report = await evaluator.generate_interview_report(
            candidate_name="Alex Johnson",
            role_title="Senior Backend Engineer",
            duration_minutes=45,
            evaluations=evaluations,
        )
        
        print_success("Interview report generated!")
        print(f"\n{Colors.BOLD}=== INTERVIEW REPORT ==={Colors.RESET}")
        print(f"Candidate: {report.candidate_name}")
        print(f"Role: {report.role_title}")
        print(f"Duration: {report.duration_minutes} minutes")
        print(f"\n{Colors.BOLD}Scores:{Colors.RESET}")
        print(f"  Technical: {report.technical_score:.1f}/100")
        print(f"  Behavioral: {report.behavioral_score:.1f}/100")
        print(f"  Communication: {report.communication_score:.1f}/100")
        print(f"  Problem Solving: {report.problem_solving_score:.1f}/100")
        print(f"  {Colors.BOLD}Overall: {report.overall_score:.1f}/100{Colors.RESET}")
        print(f"\n{Colors.BOLD}Recommendation: {report.recommendation.value.upper()}{Colors.RESET}")
        print(f"Confidence: {report.confidence:.0f}%")
        
        if report.executive_summary:
            print(f"\n{Colors.BOLD}Executive Summary:{Colors.RESET}")
            print(f"  {report.executive_summary[:200]}...")
        
        if report.next_steps:
            print(f"\n{Colors.BOLD}Next Steps:{Colors.RESET}")
            for step in report.next_steps[:3]:
                print(f"  • {step}")
        
        return True
        
    except Exception as e:
        print_error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     AI INTERVIEWER - LLM INTEGRATION TEST SUITE           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(Colors.RESET)
    
    results = {}
    
    # Test 1: LLM Provider Connection
    success, provider = await test_llm_provider_connection()
    results['llm_connection'] = success
    
    if not success:
        print_error("\nCannot continue without LLM connection. Exiting.")
        return results
    
    # Test 2: Question Generation
    success, questions = await test_question_generation(provider)
    results['question_generation'] = success
    
    # Test 3: Follow-up Generation
    if questions.get('technical'):
        tech_q = questions['technical'][0].question
        success = await test_follow_up_generation(provider, tech_q)
        results['follow_up'] = success
    else:
        print_warning("Skipping follow-up test (no technical questions generated)")
        results['follow_up'] = False
    
    # Test 4: Answer Evaluation
    success, evaluations = await test_answer_evaluation(provider)
    results['answer_evaluation'] = success
    
    # Test 5: Interview Report
    if evaluations:
        success = await test_interview_report(provider, evaluations)
        results['interview_report'] = success
    else:
        print_warning("Skipping report test (no evaluations available)")
        results['interview_report'] = False
    
    # Summary
    print_header("Test Results Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! LLM integration is working correctly.{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}Some tests failed. Check the output above for details.{Colors.RESET}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
