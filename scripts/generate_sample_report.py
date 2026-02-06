"""
Generate a sample PDF report for review.

This script creates a complete interview session, answers questions,
and saves a PDF report to the reports/ folder.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.interview import InterviewConfig
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo
from src.services.interview_orchestrator import InterviewOrchestrator


async def generate_sample_report():
    """Generate a sample interview report."""
    print("=" * 60)
    print("GENERATING SAMPLE PDF REPORT")
    print("=" * 60)
    
    resume = ParsedResume(
        contact=ContactInfo(name="John Smith", email="john@example.com"),
        summary="Senior Python developer with 5 years experience in backend systems. "
                "Expert in building scalable APIs and microservices.",
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis"],
        experience=[{
            "company": "TechCorp Inc.",
            "title": "Senior Developer",
            "start_date": "2020-01",
            "end_date": "Present",
            "highlights": [
                "Built scalable REST APIs serving 1M+ requests/day",
                "Led migration to microservices architecture",
                "Mentored team of 3 junior developers",
            ],
        }],
        education=[{
            "institution": "State University",
            "degree": "B.S. Computer Science",
            "end_date": "2019",
        }],
        raw_text="John Smith - Senior Python Developer",
    )
    
    jd = ParsedJobDescription(
        title="Backend Engineer",
        company="HackGeniX Tech",
        location="Remote",
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_skills=["AWS", "Docker", "Kubernetes"],
        responsibilities=[
            "Design and build backend services",
            "Write clean, tested code",
            "Collaborate with frontend team",
        ],
        raw_text="Backend Engineer at HackGeniX Tech",
    )
    
    config = InterviewConfig(
        screening_questions=2,
        technical_questions=3,
        behavioral_questions=2,
        system_design_questions=0,
        enable_follow_ups=False,
    )
    
    orchestrator = InterviewOrchestrator()
    
    print("\n[1] Starting interview session...")
    session, response = await orchestrator.start_interview(
        resume=resume,
        jd=jd,
        resume_id="demo-resume",
        jd_id="demo-jd",
        config=config,
    )
    print(f"    Session ID: {session.id}")
    print(f"    Total questions: {len(session.questions)}")
    
    # Realistic answers for each stage
    answers = {
        "screening": [
            "I'm a senior Python developer with 5 years of experience building backend systems. "
            "At TechCorp, I've built REST APIs that handle over a million requests daily. "
            "I'm passionate about clean code, testing, and mentoring junior developers.",
            
            "I'm excited about this role because HackGeniX is working on innovative AI technology. "
            "The tech stack aligns perfectly with my expertise in Python and FastAPI. "
            "I'm also drawn to the remote-first culture and growth opportunities.",
        ],
        "technical": [
            "For handling high traffic, I use several strategies: horizontal scaling with load balancers, "
            "Redis caching for frequently accessed data, database connection pooling, "
            "async I/O with FastAPI, and proper database indexing. I also implement circuit breakers "
            "for graceful degradation when dependencies fail.",
            
            "My debugging approach is systematic. First, I reproduce the issue consistently. "
            "Then I check logs and metrics to understand the timeline. For distributed systems, "
            "I use tracing tools like Jaeger. I isolate components and test individually. "
            "Once found, I write a failing test, fix the issue, and document the root cause.",
            
            "I follow TDD principles where practical. I write unit tests for business logic, "
            "integration tests for API endpoints, and end-to-end tests for critical flows. "
            "I aim for 80% coverage on core code. I use pytest with fixtures and mock external services. "
            "Tests run in CI/CD pipeline before any merge.",
        ],
        "behavioral": [
            "I had a teammate who was resistant to code reviews. I scheduled a private conversation "
            "to understand their perspective. They felt their experience wasn't valued. "
            "I started highlighting positives in reviews first and framed suggestions as questions. "
            "We established team guidelines together, which improved everyone's experience.",
            
            "We had a critical feature needed in 2 weeks instead of 6. I met with stakeholders "
            "to understand core requirements and negotiated scope reduction. I broke down tasks, "
            "communicated progress daily, and identified blockers early. We delivered core functionality "
            "on time and added remaining features in the next sprint.",
        ],
        "wrap_up": [
            "I'd like to know more about the team structure and development process. "
            "How does the team handle technical debt? What are the growth opportunities? "
            "I'm very excited about contributing to this AI interview product.",
        ],
    }
    
    print("\n[2] Answering interview questions...")
    answer_idx = {"screening": 0, "technical": 0, "behavioral": 0, "wrap_up": 0}
    q_num = 0
    
    while True:
        current_q = session.current_question
        if not current_q:
            break
        
        stage = current_q.stage if isinstance(current_q.stage, str) else current_q.stage.value
        stage_answers = answers.get(stage, answers["screening"])
        idx = answer_idx.get(stage, 0) % len(stage_answers)
        answer_text = stage_answers[idx]
        answer_idx[stage] = idx + 1
        
        submit_response = await orchestrator.submit_answer(
            session_id=session.id,
            answer_text=answer_text,
        )
        
        q_num += 1
        score = submit_response.evaluation.get("scores", {}).get("overall", 0) if submit_response.evaluation else 0
        print(f"    Q{q_num}: [{stage.upper():10}] Score: {score:.1f}")
        
        if submit_response.interview_complete:
            break
        
        session = orchestrator.get_session(session.id)
    
    print(f"\n    Completed {q_num} questions")
    
    # Generate JSON report
    print("\n[3] Generating report...")
    report = await orchestrator.generate_report(session.id)
    
    print(f"\n    === REPORT SUMMARY ===")
    print(f"    Candidate: {report.candidate_name}")
    print(f"    Role: {report.role_title}")
    print(f"    Overall Score: {report.overall_score:.1f}/100")
    print(f"    Technical Score: {report.technical_score:.1f}/100")
    print(f"    Behavioral Score: {report.behavioral_score:.1f}/100")
    print(f"    Communication: {report.communication_score:.1f}/100")
    print(f"    Recommendation: {report.recommendation.upper()}")
    print(f"    Confidence: {report.confidence:.0f}%")
    
    # Generate PDF
    print("\n[4] Generating PDF report...")
    pdf_bytes = await orchestrator.export_pdf(session.id)
    
    # Save to reports folder
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / f"sample_report_{session.id[:8]}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    
    print(f"\n    PDF saved to: {pdf_path}")
    print(f"    PDF size: {len(pdf_bytes):,} bytes ({len(pdf_bytes)/1024:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("SAMPLE REPORT GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOpen the PDF at:\n  {pdf_path.absolute()}")
    
    return pdf_path


if __name__ == "__main__":
    asyncio.run(generate_sample_report())
