"""
End-to-End Interview Simulation (Text Mode).

Runs a complete interview session with:
1. Start interview with hybrid question generation
2. Answer all questions with simulated candidate responses
3. Generate final report

Prerequisites:
- Ollama running with qwen2.5:3b model

Usage:
    .venv/Scripts/python.exe scripts/run_interview_e2e.py
"""
import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.interview import (
    InterviewConfig,
    InterviewStatus,
    InterviewStage,
    InterviewMode,
)
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo, Experience, Education
from src.services.interview_orchestrator import InterviewOrchestrator


# Simulated candidate answers for different question types
SIMULATED_ANSWERS = {
    "screening": [
        """I'm Alex Thompson, a senior backend engineer with 7 years of experience. 
        I started my career at a fintech startup where I built payment processing systems,
        then moved to MegaTech where I've been leading the platform services team.
        I'm particularly passionate about distributed systems and event-driven architectures.
        What excites me about this role is the opportunity to work on scalable infrastructure
        and mentor other engineers.""",
        
        """I'm interested in this position because it aligns perfectly with my experience
        in building distributed systems. I've been following your company's work on 
        real-time data processing and I'm excited about the technical challenges involved.
        The focus on system design and mentorship also matches my career goals.""",
        
        """My key technical strengths are Python and Go for backend development,
        designing scalable microservices architectures, and working with event-driven
        systems using Kafka. I also have strong experience with cloud infrastructure
        on AWS and container orchestration with Kubernetes.""",
    ],
    
    "technical": [
        """For microservices architecture, I follow domain-driven design principles.
        Each service owns its data and communicates via well-defined APIs or events.
        I use API gateways for routing, implement circuit breakers for resilience,
        and ensure observability with distributed tracing. Key considerations include
        service boundaries, data consistency patterns like saga, and deployment strategies.""",
        
        """When debugging distributed systems, I start with centralized logging and
        tracing using tools like Jaeger or Zipkin. I look at correlation IDs to trace
        requests across services. For performance issues, I use profiling tools and
        analyze metrics like latency percentiles, error rates, and throughput.
        I also use chaos engineering principles to identify failure modes proactively.""",
        
        """For database scaling, I consider read replicas for read-heavy workloads,
        sharding for horizontal scaling, and caching layers with Redis. I've implemented
        connection pooling, query optimization, and proper indexing strategies.
        For write-heavy scenarios, I've used event sourcing and CQRS patterns.""",
        
        """API design best practices I follow include RESTful principles with proper
        HTTP methods and status codes, versioning strategies, pagination for large datasets,
        rate limiting, and comprehensive documentation with OpenAPI. I also implement
        proper error handling with meaningful error messages and codes.""",
        
        """For caching strategies, I use multi-level caching - application-level with
        in-memory caches, distributed caching with Redis, and CDN for static content.
        I implement cache invalidation patterns like TTL, write-through, and cache-aside.
        Key considerations include cache stampede prevention and maintaining consistency.""",
    ],
    
    "behavioral": [
        """In my previous role, I had a disagreement with a senior architect about 
        microservices boundaries. I scheduled a one-on-one to understand their perspective,
        presented data from our system metrics, and we ultimately found a compromise
        that addressed both scalability and team ownership concerns. The result was
        a clearer service boundary that reduced cross-team dependencies by 40%.""",
        
        """When we had a critical deadline for a payment system migration, I broke down
        the work into phases, identified the critical path, and coordinated with 
        three teams. I implemented daily standups and created a shared dashboard for
        progress tracking. We delivered on time by parallelizing work and cutting
        non-essential features for a later release.""",
        
        """I mentored a junior engineer who was struggling with system design concepts.
        I created a structured learning plan, paired with them on real projects,
        and gave regular feedback. Within 6 months, they successfully led their first
        service design and are now mentoring others themselves.""",
    ],
    
    "system_design": [
        """For a URL shortener at scale, I'd design it with:
        
        1. API Layer: REST endpoints for create/redirect, behind a load balancer
        2. Short URL Generation: Base62 encoding with a distributed ID generator (Snowflake-like)
        3. Storage: Primary database (PostgreSQL) with Redis cache for hot URLs
        4. Redirection: Cache lookup first, then DB, with 301 redirects
        5. Analytics: Async event stream to Kafka for click tracking
        
        For scaling to millions of URLs:
        - Horizontal scaling of API servers
        - Database sharding by URL hash
        - Multi-region deployment with geo-routing
        - Rate limiting to prevent abuse
        
        Trade-offs: I'd start with a simpler architecture and add complexity as needed,
        monitoring for bottlenecks and scaling specific components.""",
    ],
    
    "wrap_up": [
        """I'd like to know more about the team structure and how engineering decisions
        are made. Also, what does the onboarding process look like, and what would
        success look like in the first 90 days? I'm also curious about the tech stack
        evolution plans and opportunities for technical leadership.""",
    ],
}


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
            Experience(
                company="MegaTech Corp",
                title="Senior Backend Engineer",
                location="San Francisco, CA",
                start_date="2020-03",
                end_date="Present",
                description="Lead backend engineer for core platform services.",
                highlights=[
                    "Architected event-driven microservices handling 50K events/sec",
                    "Designed and implemented real-time analytics pipeline with Kafka",
                    "Led migration from AWS to multi-cloud setup",
                ],
            ),
            Experience(
                company="StartupABC",
                title="Software Engineer",
                location="New York, NY",
                start_date="2017-01",
                end_date="2020-02",
                description="Backend developer for fintech platform.",
                highlights=[
                    "Built payment processing service handling $1M+ daily transactions",
                    "Implemented caching layer reducing API latency by 75%",
                ],
            ),
        ],
        education=[
            Education(
                institution="Stanford University",
                degree="M.S. Computer Science",
                field="Computer Science",
                end_date="2016",
            ),
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


def get_simulated_answer(stage: str, question_index: int) -> str:
    """Get a simulated answer for a question."""
    stage_key = stage.lower().replace("_", "")
    if stage_key == "systemdesign":
        stage_key = "system_design"
    elif stage_key == "wrapup":
        stage_key = "wrap_up"
    
    answers = SIMULATED_ANSWERS.get(stage_key, SIMULATED_ANSWERS["technical"])
    return answers[question_index % len(answers)]


def print_separator(title: str, char: str = "="):
    """Print a section separator."""
    print(f"\n{char * 70}")
    print(f" {title}")
    print(char * 70)


def print_question(question, index: int):
    """Print a question."""
    source = getattr(question, 'source', 'generated')
    print(f"\n  Q{index}: {question.question_text}")
    print(f"      [Stage: {question.stage} | Difficulty: {question.difficulty} | Source: {source}]")


def print_evaluation(evaluation: dict):
    """Print evaluation summary."""
    scores = evaluation.get("scores", {})
    print(f"\n  Evaluation:")
    print(f"    - Overall Score: {scores.get('overall', 0):.1f}/100")
    print(f"    - Technical Accuracy: {scores.get('technical_accuracy', 0):.1f}")
    print(f"    - Completeness: {scores.get('completeness', 0):.1f}")
    print(f"    - Clarity: {scores.get('clarity', 0):.1f}")
    print(f"    - Recommendation: {evaluation.get('recommendation', 'N/A')}")
    
    strengths = evaluation.get("strengths", [])
    if strengths:
        print(f"    - Strengths: {', '.join(strengths[:2])}")
    
    improvements = evaluation.get("improvements", [])
    if improvements:
        print(f"    - Areas to improve: {', '.join(improvements[:2])}")


async def run_interview():
    """Run a complete interview session."""
    print_separator("AI INTERVIEW SIMULATION - E2E TEST")
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Mode: Text (Audio bypassed)")
    print("  Question Generation: Hybrid (Bank + LLM)")
    
    # Create test data
    resume = create_test_resume()
    jd = create_test_jd()
    
    print(f"\n  Candidate: {resume.contact.name}")
    print(f"  Position: {jd.title} at {jd.company}")
    
    # Configure interview
    config = InterviewConfig(
        mode=InterviewMode.TEXT,
        screening_questions=2,
        technical_questions=3,
        behavioral_questions=2,
        system_design_questions=1,
        adaptive_difficulty=True,
        enable_follow_ups=False,  # Disable for faster testing
        use_question_bank=True,
        auto_detect_domains=True,
        bank_question_ratio=0.6,
        allow_rephrasing=True,
        allow_personalization=True,
    )
    
    # Initialize orchestrator
    orchestrator = InterviewOrchestrator()
    
    # Start interview
    print_separator("STARTING INTERVIEW", "-")
    start_time = time.time()
    
    try:
        session, start_response = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-resume-001",
            jd_id="test-jd-001",
            config=config,
        )
    except Exception as e:
        print(f"\n  ERROR starting interview: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    init_time = time.time() - start_time
    
    print(f"\n  Session ID: {session.id}")
    print(f"  Status: {start_response.status}")
    print(f"  Total Questions: {start_response.total_questions}")
    print(f"  Initialization Time: {init_time:.2f}s")
    
    # Show question distribution by source
    source_counts = {}
    for q in session.questions:
        source = getattr(q, 'source', 'generated')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n  Question Sources:")
    for source, count in source_counts.items():
        print(f"    - {source}: {count}")
    
    # Answer each question
    print_separator("INTERVIEW IN PROGRESS", "-")
    
    question_num = 0
    current_stage = None
    stage_question_index = {}
    
    while session.status == InterviewStatus.IN_PROGRESS:
        question = session.current_question
        if not question:
            break
        
        question_num += 1
        stage = question.stage
        
        # Track question index per stage
        if stage not in stage_question_index:
            stage_question_index[stage] = 0
        else:
            stage_question_index[stage] += 1
        
        # Print stage header if changed
        if stage != current_stage:
            current_stage = stage
            print_separator(f"STAGE: {stage.upper()}", "-")
        
        # Print question
        print_question(question, question_num)
        
        # Get simulated answer
        answer = get_simulated_answer(stage, stage_question_index[stage])
        print(f"\n  Candidate's Answer (simulated):")
        print(f"    \"{answer[:150]}...\"" if len(answer) > 150 else f"    \"{answer}\"")
        
        # Submit answer
        try:
            response = await orchestrator.submit_answer(
                session_id=session.id,
                answer_text=answer,
            )
        except Exception as e:
            print(f"\n  ERROR submitting answer: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Print evaluation
        if response.evaluation:
            print_evaluation(response.evaluation)
        
        # Show progress
        print(f"\n  Progress: {response.questions_answered}/{response.total_questions} ({response.progress_percent:.0f}%)")
        
        if response.stage_changed:
            print(f"  >> Stage transition to: {response.current_stage}")
        
        if response.interview_complete:
            print("\n  >> Interview Complete!")
            break
        
        # Small delay to avoid overwhelming the LLM
        await asyncio.sleep(0.5)
    
    # Generate final report
    print_separator("GENERATING FINAL REPORT", "-")
    
    try:
        report = await orchestrator.generate_report(session.id)
    except Exception as e:
        print(f"\n  ERROR generating report: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    total_time = time.time() - start_time
    
    # Print report summary
    print_separator("INTERVIEW REPORT", "=")
    
    print(f"\n  Candidate: {report.candidate_name}")
    print(f"  Position: {report.role_title}")
    print(f"  Duration: {report.duration_minutes} minutes")
    
    print(f"\n  SCORES:")
    print(f"    - Overall:       {report.overall_score:.1f}/100")
    print(f"    - Technical:     {report.technical_score:.1f}/100")
    print(f"    - Behavioral:    {report.behavioral_score:.1f}/100")
    print(f"    - Communication: {report.communication_score:.1f}/100")
    
    print(f"\n  EXECUTIVE SUMMARY:")
    print(f"    {report.executive_summary}")
    
    print(f"\n  RECOMMENDATION: {report.recommendation.upper()}")
    print(f"  Confidence: {report.confidence:.0f}%")
    
    if report.reasoning:
        print(f"\n  REASONING:")
        print(f"    {report.reasoning[:300]}...")
    
    if report.strengths:
        print(f"\n  KEY STRENGTHS:")
        for i, s in enumerate(report.strengths[:3], 1):
            if isinstance(s, dict):
                print(f"    {i}. {s.get('area', s.get('title', 'Strength'))}: {s.get('evidence', s.get('description', ''))[:80]}")
            else:
                print(f"    {i}. {s}")
    
    if report.concerns:
        print(f"\n  AREAS OF CONCERN:")
        for i, c in enumerate(report.concerns[:3], 1):
            if isinstance(c, dict):
                print(f"    {i}. {c.get('area', c.get('title', 'Concern'))}: {c.get('evidence', c.get('description', ''))[:80]}")
            else:
                print(f"    {i}. {c}")
    
    if report.next_steps:
        print(f"\n  RECOMMENDED NEXT STEPS:")
        for i, step in enumerate(report.next_steps[:3], 1):
            print(f"    {i}. {step}")
    
    # Export PDF report
    print_separator("EXPORTING PDF REPORT", "-")
    
    try:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = reports_dir / f"interview_report_{timestamp}.pdf"
        
        pdf_bytes = await orchestrator.export_pdf(session.id, str(pdf_path))
        print(f"\n  PDF Report saved to: {pdf_path}")
        print(f"  File size: {len(pdf_bytes) / 1024:.1f} KB")
    except Exception as e:
        print(f"\n  WARNING: Could not generate PDF: {e}")
    
    # Final summary
    print_separator("TEST SUMMARY", "=")
    print(f"\n  Total Execution Time: {total_time:.1f}s")
    print(f"  Questions Answered: {len(session.answers)}")
    print(f"  Final Status: {session.status.value if hasattr(session.status, 'value') else session.status}")
    print(f"\n  Result: SUCCESS")
    
    return True


async def main():
    """Main entry point."""
    try:
        success = await run_interview()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n  Interview cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
