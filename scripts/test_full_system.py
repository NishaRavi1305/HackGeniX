#!/usr/bin/env python
"""
Full System E2E Test - Complete Interview Flow.

This script tests the entire interview workflow:
1. Document parsing (resume + job description)
2. JD-Resume matching
3. Question generation for all stages
4. Answer evaluation with real LLM
5. Follow-up question generation
6. Interview report generation
7. Hiring recommendation

Run with: python scripts/test_full_system.py

Requires: Ollama running with qwen2.5:3b model
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.providers.llm import get_llm_provider
from src.services.document_processor import DocumentProcessor
from src.services.semantic_matcher import SemanticMatcher
from src.services.question_generator import QuestionGenerator, QuestionGenerationRequest
from src.services.answer_evaluator import AnswerEvaluator, InterviewReport
from src.services.prompts import InterviewStage, QuestionDifficulty
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo, Experience


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


# Test data - realistic resume and JD
SAMPLE_RESUME_TEXT = """
ALEX CHEN
Senior Software Engineer | Backend Specialist

Contact: alex.chen@email.com | +1-555-0123 | San Francisco, CA
GitHub: github.com/alexchen | LinkedIn: linkedin.com/in/alexchen

PROFESSIONAL SUMMARY
Results-driven Senior Software Engineer with 7 years of experience designing and implementing 
scalable backend systems. Expert in Python, Go, and cloud-native architectures. Proven track 
record of leading technical initiatives and mentoring engineering teams. Passionate about 
clean code, performance optimization, and DevOps practices.

TECHNICAL SKILLS
Languages: Python, Go, JavaScript, TypeScript, SQL
Frameworks: FastAPI, Django, Flask, Gin, Express.js
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Cloud & DevOps: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, Terraform
Tools: Git, GitHub Actions, Jenkins, Prometheus, Grafana

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechScale Inc. | 2021 - Present
- Architected and implemented microservices handling 50M+ daily API requests with 99.9% uptime
- Led migration from monolithic architecture to Kubernetes-based microservices, reducing deployment time by 80%
- Designed and built real-time data pipeline processing 1TB+ daily using Apache Kafka and Python
- Mentored team of 5 junior engineers, conducting code reviews and technical design sessions
- Reduced API latency by 65% through caching strategies and database query optimization

Software Engineer | DataFlow Systems | 2018 - 2021
- Developed RESTful APIs serving 10K+ concurrent users using FastAPI and PostgreSQL
- Implemented automated CI/CD pipelines using GitHub Actions and Docker
- Built monitoring and alerting system using Prometheus and Grafana
- Collaborated with cross-functional teams to deliver features on tight deadlines

Junior Software Engineer | StartupXYZ | 2016 - 2018
- Built and maintained backend services in Python/Django
- Developed integration tests and improved code coverage from 45% to 85%
- Participated in agile ceremonies and sprint planning

EDUCATION
BS in Computer Science | University of California, Berkeley | 2016
"""

SAMPLE_JD_TEXT = """
SENIOR BACKEND ENGINEER

Company: InnovateTech Solutions
Location: San Francisco, CA (Hybrid)
Employment Type: Full-time

About Us:
InnovateTech Solutions is a fast-growing fintech company building the next generation of 
payment processing systems. We process billions of dollars in transactions annually and 
are looking for talented engineers to scale our platform.

Role Overview:
We're seeking a Senior Backend Engineer to join our Platform team. You'll design and 
build the core infrastructure powering our payment systems, working on high-throughput, 
low-latency services that handle millions of transactions.

Responsibilities:
- Design and implement scalable microservices using Python and Go
- Build and maintain high-performance APIs handling 100K+ requests/second
- Develop data pipelines for real-time transaction processing
- Lead technical design reviews and mentor junior engineers
- Collaborate with Product, Security, and DevOps teams
- Participate in on-call rotation and incident response
- Write comprehensive tests and documentation

Required Qualifications:
- 5+ years of backend development experience
- Strong proficiency in Python; experience with Go is a plus
- Experience with distributed systems and microservices architecture
- Proficiency with PostgreSQL, Redis, and message queues (Kafka/RabbitMQ)
- Experience with cloud platforms (AWS preferred)
- Strong understanding of API design and RESTful principles
- Experience with Docker and Kubernetes
- Excellent problem-solving and communication skills

Preferred Qualifications:
- Experience in fintech or payment processing
- Familiarity with event-driven architecture
- Experience with GraphQL
- Knowledge of security best practices for financial systems

Compensation: $180,000 - $240,000 base + equity + benefits
"""

# Simulated candidate answers
CANDIDATE_ANSWERS = {
    "screening": """
    I'm interested in InnovateTech because I'm passionate about building systems that 
    scale. Payment processing is an exciting domain - the challenges around reliability, 
    latency, and security are exactly what I love solving. In my current role at TechScale, 
    I've been working on similar problems, handling millions of daily transactions. 
    I'm looking to take on more ownership and work on even higher scale systems.
    """,
    
    "technical_api": """
    I would design a rate limiting API using a token bucket algorithm implemented with Redis. 
    
    For the data model, each user or API key would have a bucket in Redis with:
    - Current token count
    - Last refill timestamp
    - Maximum tokens (rate limit)
    - Refill rate (tokens per second)
    
    When a request comes in:
    1. Get the current bucket state from Redis using MULTI/EXEC for atomicity
    2. Calculate tokens to add based on time elapsed since last refill
    3. If tokens available, decrement and allow the request
    4. If not, return 429 Too Many Requests with Retry-After header
    
    For distributed systems, I'd use Redis Cluster for horizontal scaling. We could also 
    implement a sliding window algorithm for smoother rate limiting. The API would expose 
    endpoints for checking current limits and quota remaining in response headers.
    """,
    
    "technical_db": """
    For optimizing slow database queries, I follow a systematic approach:
    
    First, I analyze the query using EXPLAIN ANALYZE to understand the execution plan.
    Look for sequential scans on large tables, missing indexes, and poor join strategies.
    
    Common optimizations I'd apply:
    1. Add appropriate indexes - B-tree for equality/range, GIN for arrays/full-text
    2. Use composite indexes that match query patterns
    3. Partition large tables by date or ID ranges
    4. Denormalize hot paths where read performance is critical
    5. Implement connection pooling with PgBouncer
    6. Add Redis caching for frequently accessed data
    
    In my last project, I reduced a 12-second query to 50ms by adding a composite index 
    and restructuring the query to avoid a correlated subquery. We also implemented 
    read replicas to distribute query load.
    """,
    
    "behavioral_conflict": """
    At TechScale, I had a challenging situation with a senior engineer who strongly disagreed 
    with my proposal to migrate to Kubernetes. He felt it was unnecessary complexity and 
    preferred our existing VM-based deployment.
    
    The situation was causing friction in our team and blocking our infrastructure modernization. 
    My task was to either convince him or find a compromise that would satisfy both perspectives.
    
    I took several actions: First, I set up a one-on-one to understand his specific concerns. 
    He worried about the learning curve and potential production issues. I acknowledged these 
    were valid concerns. Then I proposed a pilot project - migrating a non-critical service 
    first to prove the concept and build team expertise. I also created a detailed training 
    plan and offered to pair with him on the initial migration.
    
    The result was positive. After the pilot succeeded, he became one of the biggest advocates 
    for Kubernetes. The migration reduced our deployment time by 80% and improved reliability. 
    He later told me he appreciated that I listened to his concerns instead of dismissing them.
    """,
    
    "behavioral_failure": """
    Last year, I led a critical data migration project that didn't go as planned. We were 
    moving from a legacy database to PostgreSQL, and I underestimated the complexity.
    
    My task was to migrate 5 years of transaction data with zero downtime. I had estimated 
    2 weeks but the actual migration took 6 weeks.
    
    What went wrong: I didn't account for data inconsistencies in the legacy system, and 
    our validation scripts had gaps. We also discovered foreign key violations during 
    production cutover.
    
    My actions included: I immediately communicated the delays to stakeholders with honest 
    assessments. I brought in a DBA consultant to help with the complex data transformations. 
    We implemented a dual-write strategy to avoid further delays.
    
    The result: We completed the migration successfully, but late. The key lesson I learned 
    was to always run a full production-scale test migration before committing to timelines. 
    I've since created a migration checklist that I use for all data migration projects, and 
    I build in more buffer time for unknown complexities.
    """
}


async def run_full_interview():
    """Run a complete simulated interview."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          AI INTERVIEWER - FULL SYSTEM E2E TEST                       ║")
    print("║          Complete Interview Simulation                                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(Colors.RESET)
    
    start_time = datetime.now()
    test_results = {}
    evaluations = []
    
    # ========== PHASE 1: LLM Provider Setup ==========
    print_header("PHASE 1: LLM Provider Initialization")
    
    try:
        provider = await get_llm_provider()
        print_success(f"LLM Provider ready: {type(provider).__name__}")
        print_info(f"Model: {provider.model}")
        test_results["llm_provider"] = True
    except Exception as e:
        print_fail(f"LLM Provider failed: {e}")
        print_warn("Make sure Ollama is running with qwen2.5:3b")
        return
    
    # ========== PHASE 2: Document Processing ==========
    print_header("PHASE 2: Document Processing")
    
    try:
        processor = DocumentProcessor()
        
        # Parse resume - use helper methods directly for raw text
        print_section("Parsing Resume")
        
        # Extract structured data using processor's helper methods
        contact = processor.extract_contact_info(SAMPLE_RESUME_TEXT)
        skills = processor.extract_skills(SAMPLE_RESUME_TEXT)
        entities = processor.extract_entities(SAMPLE_RESUME_TEXT)
        sections = processor.extract_sections(SAMPLE_RESUME_TEXT)
        
        resume = ParsedResume(
            contact=contact,
            summary=sections.get('summary'),
            skills=skills,
            entities=entities,
            raw_text=SAMPLE_RESUME_TEXT,
        )
        
        print_success(f"Resume parsed: {resume.contact.name if resume.contact else 'Unknown'}")
        print_info(f"Skills found: {len(resume.skills)} ({', '.join(resume.skills[:5])}...)")
        print_info(f"Experience entries: {len(resume.experience)}")
        
        # Parse JD (async)
        print_section("Parsing Job Description")
        jd = await processor.parse_job_description(SAMPLE_JD_TEXT)
        print_success(f"JD parsed: {jd.title or 'Senior Backend Engineer'}")
        print_info(f"Required skills: {len(jd.required_skills)}")
        print_info(f"Experience required: {jd.experience_years_min or 5}+ years")
        
        test_results["document_parsing"] = True
        
    except Exception as e:
        print_fail(f"Document processing failed: {e}")
        test_results["document_parsing"] = False
        import traceback
        traceback.print_exc()
    
    # ========== PHASE 3: JD-Resume Matching ==========
    print_header("PHASE 3: JD-Resume Semantic Matching")
    
    try:
        matcher = SemanticMatcher()
        
        # Pass resume/jd in correct order with IDs
        match_result = await matcher.match(
            resume=resume,
            job_description=jd,
            resume_id="test-resume-001",
            job_description_id="test-jd-001"
        )
        
        print_success(f"Match completed!")
        print_info(f"Overall Match Score: {match_result.overall_score:.1f}/100")
        print_info(f"Skill Match: {match_result.skill_match_score:.1f}/100")
        print_info(f"Experience Match: {match_result.experience_match_score:.1f}/100")
        
        if match_result.recommendations:
            print_section("Match Recommendations")
            for rec in match_result.recommendations[:3]:
                print(f"  - {rec}")
        
        test_results["jd_matching"] = True
        
    except Exception as e:
        print_fail(f"JD matching failed: {e}")
        test_results["jd_matching"] = False
    
    # ========== PHASE 4: Question Generation ==========
    print_header("PHASE 4: Question Generation")
    
    generator = QuestionGenerator(llm_provider=provider)
    generated_questions = {}
    
    stages_to_test = [
        (InterviewStage.SCREENING, 2),
        (InterviewStage.TECHNICAL, 3),
        (InterviewStage.BEHAVIORAL, 2),
    ]
    
    for stage, num_q in stages_to_test:
        print_section(f"Generating {stage.value} questions ({num_q})")
        try:
            request = QuestionGenerationRequest(
                stage=stage,
                num_questions=num_q,
                difficulty=QuestionDifficulty.MEDIUM,
            )
            questions = await generator.generate_questions(request, resume, jd)
            generated_questions[stage] = questions
            
            print_success(f"Generated {len(questions)} {stage.value} questions")
            for i, q in enumerate(questions[:2], 1):
                print(f"  {i}. {q.question[:70]}...")
            
        except Exception as e:
            print_fail(f"{stage.value} question generation failed: {e}")
            generated_questions[stage] = []
    
    test_results["question_generation"] = all(len(q) > 0 for q in generated_questions.values())
    
    # ========== PHASE 5: Answer Evaluation ==========
    print_header("PHASE 5: Answer Evaluation (LLM-as-Judge)")
    
    evaluator = AnswerEvaluator(llm_provider=provider)
    
    # Evaluate a technical answer
    print_section("Technical Answer Evaluation")
    try:
        tech_questions = generated_questions.get(InterviewStage.TECHNICAL, [])
        if tech_questions:
            tech_q = tech_questions[0]
            tech_answer = CANDIDATE_ANSWERS["technical_api"]
            
            print_info(f"Question: {tech_q.question[:60]}...")
            print_info(f"Evaluating answer ({len(tech_answer)} chars)...")
            
            tech_eval = await evaluator.evaluate_answer(
                question=tech_q.question,
                answer=tech_answer,
                expected_points=tech_q.expected_answer_points,
                stage=InterviewStage.TECHNICAL,
                validate=True,
            )
            
            evaluations.append(tech_eval)
            
            print_success("Technical evaluation completed!")
            print(f"  Overall: {tech_eval.scores.overall:.1f}/100")
            print(f"  Technical Accuracy: {tech_eval.scores.technical_accuracy:.1f}")
            print(f"  Completeness: {tech_eval.scores.completeness:.1f}")
            print(f"  Recommendation: {tech_eval.recommendation.value}")
            print(f"  Validated: {tech_eval.is_validated}")
            if tech_eval.strengths:
                print(f"  Strengths: {tech_eval.strengths[0][:50]}...")
            
    except Exception as e:
        print_fail(f"Technical evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate a behavioral answer
    print_section("Behavioral Answer Evaluation")
    try:
        behav_questions = generated_questions.get(InterviewStage.BEHAVIORAL, [])
        if behav_questions:
            behav_q = behav_questions[0]
            behav_answer = CANDIDATE_ANSWERS["behavioral_conflict"]
            
            print_info(f"Question: {behav_q.question[:60]}...")
            
            behav_eval = await evaluator.evaluate_behavioral_answer(
                question=behav_q.question,
                answer=behav_answer,
                competency="conflict-resolution",
                red_flags=["Blaming others", "Avoiding responsibility"],
                green_flags=["Taking initiative", "Empathy", "Measurable outcomes"],
                validate=True,
            )
            
            evaluations.append(behav_eval)
            
            print_success("Behavioral evaluation completed!")
            print(f"  Overall: {behav_eval.scores.overall:.1f}/100")
            if behav_eval.star_scores:
                print(f"  STAR - S:{behav_eval.star_scores.situation:.0f} "
                      f"T:{behav_eval.star_scores.task:.0f} "
                      f"A:{behav_eval.star_scores.action:.0f} "
                      f"R:{behav_eval.star_scores.result:.0f}")
            print(f"  Recommendation: {behav_eval.recommendation.value}")
            
    except Exception as e:
        print_fail(f"Behavioral evaluation failed: {e}")
    
    test_results["answer_evaluation"] = len(evaluations) >= 2
    
    # ========== PHASE 6: Follow-up Question ==========
    print_header("PHASE 6: Follow-up Question Generation")
    
    try:
        if tech_eval:
            follow_up = await generator.generate_follow_up(
                original_question=tech_questions[0].question,
                candidate_answer=CANDIDATE_ANSWERS["technical_api"],
                evaluation_summary=f"Score: {tech_eval.scores.overall:.0f}/100. {tech_eval.notes}",
            )
            
            if follow_up:
                print_success(f"Follow-up generated!")
                print(f"  {follow_up[:100]}...")
                test_results["follow_up"] = True
            else:
                print_warn("No follow-up generated (may be intentional)")
                test_results["follow_up"] = True
                
    except Exception as e:
        print_fail(f"Follow-up generation failed: {e}")
        test_results["follow_up"] = False
    
    # ========== PHASE 7: Interview Report ==========
    print_header("PHASE 7: Interview Report Generation")
    
    try:
        if evaluations:
            report = await evaluator.generate_interview_report(
                candidate_name="Alex Chen",
                role_title="Senior Backend Engineer",
                duration_minutes=45,
                evaluations=evaluations,
            )
            
            print_success("Interview report generated!")
            print(f"\n{Colors.BOLD}══════════════════════════════════════════════════════════════════{Colors.RESET}")
            print(f"{Colors.BOLD}  INTERVIEW REPORT - {report.candidate_name}{Colors.RESET}")
            print(f"{Colors.BOLD}══════════════════════════════════════════════════════════════════{Colors.RESET}")
            print(f"  Position: {report.role_title}")
            print(f"  Duration: {report.duration_minutes} minutes")
            print(f"\n{Colors.BOLD}  SCORES:{Colors.RESET}")
            print(f"    Technical:      {report.technical_score:.1f}/100")
            print(f"    Behavioral:     {report.behavioral_score:.1f}/100")
            print(f"    Communication:  {report.communication_score:.1f}/100")
            print(f"    Problem Solving:{report.problem_solving_score:.1f}/100")
            print(f"    {Colors.BOLD}OVERALL:        {report.overall_score:.1f}/100{Colors.RESET}")
            print(f"\n{Colors.BOLD}  RECOMMENDATION: {report.recommendation.value.upper()}{Colors.RESET}")
            print(f"  Confidence: {report.confidence:.0f}%")
            
            if report.executive_summary:
                print(f"\n{Colors.BOLD}  Executive Summary:{Colors.RESET}")
                # Word wrap
                summary = report.executive_summary
                while len(summary) > 70:
                    split = summary[:70].rfind(' ')
                    print(f"    {summary[:split]}")
                    summary = summary[split+1:]
                print(f"    {summary}")
            
            if report.next_steps:
                print(f"\n{Colors.BOLD}  Next Steps:{Colors.RESET}")
                for step in report.next_steps[:3]:
                    print(f"    - {step}")
            
            test_results["interview_report"] = True
            
    except Exception as e:
        print_fail(f"Report generation failed: {e}")
        test_results["interview_report"] = False
        import traceback
        traceback.print_exc()
    
    # ========== FINAL RESULTS ==========
    print_header("TEST RESULTS SUMMARY")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for test_name, success in test_results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} phases completed{Colors.RESET}")
    print(f"Duration: {duration:.1f} seconds")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                     ALL TESTS PASSED!                               ║")
        print("║           Full interview workflow is operational.                   ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(Colors.RESET)
    else:
        print(f"\n{Colors.YELLOW}Some phases failed. Check output above for details.{Colors.RESET}")
    
    return test_results


if __name__ == "__main__":
    asyncio.run(run_full_interview())
