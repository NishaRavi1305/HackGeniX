"""
End-to-End Test for Interview Orchestrator.

Tests a complete interview flow with real LLM calls:
1. Start interview session
2. Answer questions across all stages
3. Generate final report

Prerequisites:
- Ollama running with qwen2.5:3b model
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.interview import (
    InterviewConfig,
    InterviewStatus,
    InterviewStage,
)
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo
from src.services.interview_orchestrator import InterviewOrchestrator


def create_test_resume() -> ParsedResume:
    """Create a realistic test resume."""
    return ParsedResume(
        contact=ContactInfo(
            name="Alice Chen",
            email="alice.chen@email.com",
            phone="555-0123",
            linkedin="linkedin.com/in/alicechen",
            github="github.com/alicechen",
        ),
        summary="Senior Software Engineer with 6 years of experience in backend development. "
                "Specialized in building scalable microservices using Python, FastAPI, and cloud technologies. "
                "Passionate about clean code, testing, and mentoring junior developers.",
        skills=[
            "Python", "FastAPI", "Django", "PostgreSQL", "Redis",
            "Docker", "Kubernetes", "AWS", "Terraform",
            "GraphQL", "REST APIs", "gRPC",
            "Pytest", "CI/CD", "Git",
        ],
        experience=[
            {
                "company": "TechCorp Inc.",
                "title": "Senior Software Engineer",
                "location": "San Francisco, CA",
                "start_date": "2021-01",
                "end_date": "Present",
                "description": "Lead backend engineer for the payments platform.",
                "highlights": [
                    "Designed and implemented a high-throughput payment processing system handling 10K TPS",
                    "Led migration from monolith to microservices, reducing deployment time by 80%",
                    "Mentored 3 junior engineers and established coding standards",
                ],
            },
            {
                "company": "StartupXYZ",
                "title": "Software Engineer",
                "location": "New York, NY",
                "start_date": "2018-06",
                "end_date": "2020-12",
                "description": "Full-stack developer for e-commerce platform.",
                "highlights": [
                    "Built REST APIs serving 1M+ daily active users",
                    "Implemented caching layer that reduced API latency by 60%",
                ],
            },
        ],
        education=[
            {
                "institution": "University of California, Berkeley",
                "degree": "B.S. Computer Science",
                "field": "Computer Science",
                "end_date": "2018",
                "gpa": 3.8,
            },
        ],
        certifications=["AWS Solutions Architect Associate"],
        raw_text="Alice Chen - Senior Software Engineer with expertise in Python and cloud technologies.",
    )


def create_test_jd() -> ParsedJobDescription:
    """Create a realistic test job description."""
    return ParsedJobDescription(
        title="Senior Backend Engineer",
        company="HackGeniX Tech",
        location="Remote",
        employment_type="full-time",
        experience_level="senior",
        experience_years_min=5,
        experience_years_max=10,
        required_skills=[
            "Python", "FastAPI", "PostgreSQL", "Docker",
            "REST APIs", "Microservices", "Testing",
        ],
        preferred_skills=[
            "Kubernetes", "AWS", "Redis", "GraphQL",
            "CI/CD", "System Design",
        ],
        responsibilities=[
            "Design and implement scalable backend services",
            "Write clean, maintainable, and well-tested code",
            "Participate in code reviews and architectural discussions",
            "Mentor junior developers and promote best practices",
            "Collaborate with product and frontend teams",
        ],
        qualifications=[
            "5+ years of software engineering experience",
            "Strong proficiency in Python",
            "Experience with cloud platforms (AWS preferred)",
            "Excellent communication and collaboration skills",
        ],
        raw_text="Senior Backend Engineer position at HackGeniX Tech.",
    )


# Simulated candidate answers for different stages
CANDIDATE_ANSWERS = {
    "screening": [
        "I'm a senior software engineer with 6 years of experience, primarily focused on backend development with Python. "
        "At TechCorp, I lead the payments platform team where we process thousands of transactions per second. "
        "I'm passionate about clean code, testing, and helping junior developers grow.",
        
        "I'm excited about this role because HackGeniX is working on interesting AI interview technology. "
        "I've always been interested in developer tools and AI applications. "
        "The tech stack aligns perfectly with my experience in Python and FastAPI.",
        
        "My key technical strengths are in backend system design, API development, and cloud infrastructure. "
        "I've designed systems handling high traffic and led a major migration to microservices. "
        "I'm also strong in testing and establishing code quality practices.",
    ],
    "technical": [
        "To handle 10K requests per second, I'd use several strategies. First, implement connection pooling "
        "for database connections and use async I/O with FastAPI. Add Redis caching for frequently accessed data. "
        "Use database indexes and query optimization. Deploy behind a load balancer with horizontal scaling. "
        "Finally, implement circuit breakers to handle failures gracefully.",
        
        "For debugging complex issues, I follow a systematic approach. First, reproduce the issue consistently. "
        "Check logs and metrics to understand the timeline. Use tracing tools like Jaeger for distributed systems. "
        "Isolate variables by testing components individually. Once found, write a test case that fails, "
        "then fix and verify. Document the root cause and solution for future reference.",
        
        "I designed a payment processing system at TechCorp. It uses event-driven architecture with Kafka for reliability. "
        "Services are stateless for easy scaling. We use saga pattern for distributed transactions. "
        "PostgreSQL for transactional data, Redis for caching. Everything runs on Kubernetes with auto-scaling. "
        "The system handles failures through retries with exponential backoff and dead letter queues.",
        
        "My testing approach includes unit tests for individual functions, integration tests for API endpoints, "
        "and end-to-end tests for critical flows. I aim for 80%+ coverage on business logic. "
        "I use pytest with fixtures, mock external services, and run tests in CI/CD pipeline. "
        "I also do load testing before major releases using locust.",
        
        "I stay current through multiple channels. I follow tech blogs and newsletters like Python Weekly. "
        "I contribute to open source projects when time permits. I attend local meetups and conferences. "
        "At work, we have knowledge sharing sessions. I also experiment with new technologies in side projects.",
    ],
    "behavioral": [
        "I had a teammate who was defensive about code reviews. I scheduled a 1-on-1 to understand their perspective. "
        "Turns out they felt their experience wasn't valued. I started with more positive feedback in reviews "
        "and explained my suggestions as questions rather than demands. We established team code review guidelines "
        "together, which improved everyone's experience.",
        
        "We had a critical feature needed in 2 weeks instead of the planned 6. I met with stakeholders to understand "
        "the core requirements and negotiated scope reduction. Then I broke down tasks and worked extra hours. "
        "I communicated progress daily and identified blockers early. We delivered the core functionality on time "
        "and added remaining features in the next sprint.",
        
        "When we adopted Kubernetes, I had to learn it quickly. I started with official docs and tutorials. "
        "Set up a local cluster using minikind for hands-on practice. Read our existing configs and asked questions. "
        "Within 2 weeks I was deploying services. Within a month I was helping others. "
        "I documented my learning for the team.",
    ],
    "wrap_up": [
        "I'd like to know more about the team structure and how this role fits. "
        "What's the typical development process like? How does the team handle technical debt? "
        "Also curious about growth opportunities and learning budget. "
        "I'm very excited about the AI interviewer product - it's solving a real problem.",
    ],
}


async def run_interview_e2e_test():
    """Run a complete interview E2E test."""
    print("=" * 60)
    print("INTERVIEW ORCHESTRATOR E2E TEST")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = InterviewOrchestrator()
    
    # Create test data
    resume = create_test_resume()
    jd = create_test_jd()
    
    # Configure a shorter interview for testing
    config = InterviewConfig(
        screening_questions=2,
        technical_questions=3,
        behavioral_questions=2,
        system_design_questions=0,  # Skip for faster testing
        enable_follow_ups=False,
    )
    
    passed = 0
    failed = 0
    
    try:
        # Test 1: Start Interview
        print("\n[TEST 1] Starting Interview Session...")
        session, response = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-resume-1",
            jd_id="test-jd-1",
            config=config,
        )
        
        assert session.id is not None
        assert session.status == InterviewStatus.IN_PROGRESS or session.status == "in_progress"
        assert response.first_question is not None
        print(f"   Session ID: {session.id}")
        print(f"   Total Questions: {len(session.questions)}")
        print(f"   First Question: {response.first_question.question_text[:80]}...")
        print("   [PASSED] Interview started successfully")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Failed to start interview: {e}")
        failed += 1
        return passed, failed
    
    # Test 2: Answer Questions
    print("\n[TEST 2] Answering Interview Questions...")
    
    try:
        answer_index = {"screening": 0, "technical": 0, "behavioral": 0, "wrap_up": 0}
        questions_answered = 0
        
        while True:
            current_q = session.current_question
            if not current_q:
                break
            
            # Get stage-appropriate answer
            stage = current_q.stage if isinstance(current_q.stage, str) else current_q.stage.value
            answers = CANDIDATE_ANSWERS.get(stage, CANDIDATE_ANSWERS["screening"])
            answer_idx = answer_index.get(stage, 0) % len(answers)
            answer_text = answers[answer_idx]
            answer_index[stage] = answer_idx + 1
            
            # Submit answer
            submit_response = await orchestrator.submit_answer(
                session_id=session.id,
                answer_text=answer_text,
            )
            
            questions_answered += 1
            score = submit_response.evaluation.get("scores", {}).get("overall", 0) if submit_response.evaluation else 0
            
            print(f"   Q{questions_answered}: [{stage.upper()}] Score: {score:.1f}")
            
            if submit_response.stage_changed:
                print(f"   -> Stage changed to: {submit_response.current_stage}")
            
            if submit_response.interview_complete:
                print(f"   Interview complete after {questions_answered} questions")
                break
            
            # Update session reference
            session = orchestrator.get_session(session.id)
        
        assert questions_answered > 0
        print(f"   Answered {questions_answered} questions")
        print("   [PASSED] All questions answered successfully")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Failed during Q&A: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # Test 3: Get Progress
    print("\n[TEST 3] Checking Interview Progress...")
    try:
        progress = orchestrator.get_progress(session.id)
        
        assert progress.questions_answered > 0
        assert progress.progress_percent > 0
        print(f"   Questions: {progress.questions_answered}/{progress.total_questions}")
        print(f"   Progress: {progress.progress_percent:.1f}%")
        print(f"   Current Score: {progress.current_score:.1f}")
        print("   [PASSED] Progress retrieved successfully")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Failed to get progress: {e}")
        failed += 1
    
    # Test 4: Generate Report
    print("\n[TEST 4] Generating Interview Report...")
    try:
        report = await orchestrator.generate_report(session.id)
        
        assert report.session_id == session.id
        assert report.overall_score >= 0
        assert report.recommendation in ["strong_hire", "hire", "no_hire", "strong_no_hire"]
        
        print(f"   Candidate: {report.candidate_name}")
        print(f"   Role: {report.role_title}")
        print(f"   Overall Score: {report.overall_score:.1f}/100")
        print(f"   Technical Score: {report.technical_score:.1f}/100")
        print(f"   Behavioral Score: {report.behavioral_score:.1f}/100")
        print(f"   Recommendation: {report.recommendation.upper()}")
        print(f"   Confidence: {report.confidence:.0f}%")
        if report.executive_summary:
            print(f"   Summary: {report.executive_summary[:100]}...")
        print("   [PASSED] Report generated successfully")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # Test 5: Pause/Resume
    print("\n[TEST 5] Testing Pause/Resume...")
    try:
        # Create a new session for pause test
        session2, _ = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-resume-2",
            jd_id="test-jd-2",
            config=config,
        )
        
        paused = orchestrator.pause_interview(session2.id)
        assert paused.status == InterviewStatus.PAUSED or paused.status == "paused"
        
        resumed = orchestrator.resume_interview(session2.id)
        assert resumed.status == InterviewStatus.IN_PROGRESS or resumed.status == "in_progress"
        
        # Cleanup
        orchestrator.delete_session(session2.id)
        
        print("   [PASSED] Pause/Resume works correctly")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Pause/Resume failed: {e}")
        failed += 1
    
    # Test 6: List Sessions
    print("\n[TEST 6] Testing Session Management...")
    try:
        sessions = orchestrator.list_sessions()
        assert len(sessions) >= 1  # At least our main session
        
        deleted = orchestrator.delete_session(session.id)
        assert deleted is True
        
        sessions_after = orchestrator.list_sessions()
        assert session.id not in [s.id for s in sessions_after]
        
        print(f"   Listed {len(sessions)} sessions")
        print("   [PASSED] Session management works correctly")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] Session management failed: {e}")
        failed += 1
    
    return passed, failed


async def main():
    """Main entry point."""
    print("\nStarting Interview Orchestrator E2E Test...")
    print("This test requires Ollama with qwen2.5:3b model\n")
    
    try:
        passed, failed = await run_interview_e2e_test()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        passed, failed = 0, 1
    
    print("\n" + "=" * 60)
    print("E2E TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print("=" * 60)
    
    if failed > 0:
        print("\nSome tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
