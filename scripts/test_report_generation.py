"""
End-to-End Test for PDF Report Generation.

Tests the complete report generation flow:
1. Start a mock interview session
2. Submit answers
3. End interview and generate report
4. Request JSON report
5. Request PDF report
6. Verify PDF file exists and is valid
7. Measure generation time (must be <30s per PRD)

Prerequisites:
- Ollama running with qwen2.5:3b model
- reportlab installed
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.interview import (
    InterviewConfig,
    InterviewStatus,
)
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo
from src.services.interview_orchestrator import InterviewOrchestrator
from src.providers.report_storage import get_report_storage_provider


def create_test_resume() -> ParsedResume:
    """Create a realistic test resume."""
    return ParsedResume(
        contact=ContactInfo(
            name="John Smith",
            email="john.smith@example.com",
            phone="555-1234",
            linkedin="linkedin.com/in/johnsmith",
            github="github.com/johnsmith",
        ),
        summary="Full-stack developer with 5 years of experience in Python and JavaScript. "
                "Passionate about building scalable web applications and mentoring teams.",
        skills=[
            "Python", "JavaScript", "TypeScript", "React", "FastAPI",
            "PostgreSQL", "MongoDB", "Docker", "AWS", "Git",
        ],
        experience=[
            {
                "company": "TechStartup Inc.",
                "title": "Senior Developer",
                "location": "Remote",
                "start_date": "2020-01",
                "end_date": "Present",
                "description": "Lead developer for core platform.",
                "highlights": [
                    "Built REST APIs serving 500K daily users",
                    "Reduced API latency by 40% through optimization",
                    "Mentored 2 junior developers",
                ],
            },
        ],
        education=[
            {
                "institution": "State University",
                "degree": "B.S. Computer Science",
                "field": "Computer Science",
                "end_date": "2019",
            },
        ],
        raw_text="John Smith - Full-stack developer with Python and JavaScript expertise.",
    )


def create_test_jd() -> ParsedJobDescription:
    """Create a test job description."""
    return ParsedJobDescription(
        title="Backend Engineer",
        company="HackGeniX Tech",
        location="Remote",
        employment_type="full-time",
        experience_level="mid",
        experience_years_min=3,
        experience_years_max=7,
        required_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        preferred_skills=["AWS", "Kubernetes", "Redis"],
        responsibilities=[
            "Design and implement backend services",
            "Write clean, tested code",
            "Participate in code reviews",
        ],
        qualifications=[
            "3+ years of backend development",
            "Strong Python skills",
        ],
        raw_text="Backend Engineer at HackGeniX Tech.",
    )


# Test answers for different stages
TEST_ANSWERS = {
    "screening": [
        "I'm a full-stack developer with 5 years of experience. I specialize in Python and JavaScript, "
        "and I've built several production systems. I'm passionate about clean code and mentoring others.",
        
        "I'm interested in this role because it aligns with my backend expertise and offers growth opportunities. "
        "The tech stack matches my skills perfectly.",
    ],
    "technical": [
        "For handling high traffic, I'd use horizontal scaling with load balancing, implement caching with Redis, "
        "optimize database queries with proper indexing, and use async processing for non-critical tasks.",
        
        "My debugging approach is systematic: reproduce the issue, check logs, use debuggers and profilers, "
        "isolate components, and write tests to prevent regression.",
        
        "I follow TDD principles, write unit tests for business logic, integration tests for APIs, "
        "and aim for 80% code coverage on critical paths.",
    ],
    "behavioral": [
        "When I had a conflict with a teammate over architecture decisions, I scheduled a meeting to understand "
        "their perspective, presented data supporting my approach, and we reached a compromise that worked well.",
        
        "To meet a tight deadline, I prioritized critical features, communicated daily with stakeholders, "
        "and worked extra hours while maintaining code quality.",
    ],
    "wrap_up": [
        "I'd like to know more about the team structure and the development process. "
        "What's the approach to technical debt? I'm excited about contributing to this product.",
    ],
}


async def run_report_generation_test():
    """Run the full report generation E2E test."""
    print("=" * 70)
    print("PDF REPORT GENERATION E2E TEST")
    print("=" * 70)
    
    passed = 0
    failed = 0
    orchestrator = InterviewOrchestrator()
    storage = get_report_storage_provider()
    
    # Test data
    resume = create_test_resume()
    jd = create_test_jd()
    
    # Shorter interview config for testing
    config = InterviewConfig(
        screening_questions=2,
        technical_questions=2,
        behavioral_questions=1,
        system_design_questions=0,
        enable_follow_ups=False,
    )
    
    session = None
    
    # ================================================================
    # TEST 1: Start Interview Session
    # ================================================================
    print("\n[TEST 1] Starting Interview Session...")
    try:
        session, response = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-resume-pdf",
            jd_id="test-jd-pdf",
            config=config,
        )
        
        assert session.id is not None
        assert response.first_question is not None
        print(f"   Session ID: {session.id}")
        print(f"   Total Questions: {len(session.questions)}")
        print("   [PASSED] Interview session started")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        failed += 1
        return passed, failed, None
    
    # ================================================================
    # TEST 2: Submit Answers
    # ================================================================
    print("\n[TEST 2] Submitting Answers...")
    try:
        answer_index = {"screening": 0, "technical": 0, "behavioral": 0, "wrap_up": 0}
        questions_answered = 0
        
        while True:
            current_q = session.current_question
            if not current_q:
                break
            
            stage = current_q.stage if isinstance(current_q.stage, str) else current_q.stage.value
            answers = TEST_ANSWERS.get(stage, TEST_ANSWERS["screening"])
            answer_idx = answer_index.get(stage, 0) % len(answers)
            answer_text = answers[answer_idx]
            answer_index[stage] = answer_idx + 1
            
            submit_response = await orchestrator.submit_answer(
                session_id=session.id,
                answer_text=answer_text,
            )
            
            questions_answered += 1
            score = submit_response.evaluation.get("scores", {}).get("overall", 0) if submit_response.evaluation else 0
            print(f"   Q{questions_answered}: [{stage.upper():10}] Score: {score:.1f}")
            
            if submit_response.interview_complete:
                break
            
            session = orchestrator.get_session(session.id)
        
        assert questions_answered >= 3, f"Expected at least 3 questions, got {questions_answered}"
        print(f"   Answered {questions_answered} questions")
        print("   [PASSED] All answers submitted")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # ================================================================
    # TEST 3: Generate JSON Report
    # ================================================================
    print("\n[TEST 3] Generating JSON Report...")
    try:
        start_time = time.time()
        report = await orchestrator.generate_report(session.id)
        json_time = time.time() - start_time
        
        assert report.session_id == session.id
        assert report.overall_score >= 0
        assert report.recommendation in ["strong_hire", "hire", "no_hire", "strong_no_hire"]
        
        print(f"   Candidate: {report.candidate_name}")
        print(f"   Role: {report.role_title}")
        print(f"   Overall Score: {report.overall_score:.1f}/100")
        print(f"   Technical: {report.technical_score:.1f}, Behavioral: {report.behavioral_score:.1f}")
        print(f"   Recommendation: {report.recommendation.upper()}")
        print(f"   Generation Time: {json_time:.2f}s")
        print("   [PASSED] JSON report generated")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # ================================================================
    # TEST 4: Generate PDF Report (Performance Test)
    # ================================================================
    print("\n[TEST 4] Generating PDF Report (Performance Test)...")
    try:
        start_time = time.time()
        pdf_bytes = await orchestrator.export_pdf(session.id)
        pdf_time = time.time() - start_time
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF', "Generated file is not a valid PDF"
        
        print(f"   PDF Size: {len(pdf_bytes):,} bytes ({len(pdf_bytes)/1024:.1f} KB)")
        print(f"   Generation Time: {pdf_time:.2f}s")
        
        # PRD requirement: <30 seconds
        if pdf_time < 30:
            print(f"   [PASSED] PDF generated in <30s (requirement met)")
            passed += 1
        else:
            print(f"   [FAILED] PDF generation took {pdf_time:.1f}s (>30s requirement)")
            failed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # ================================================================
    # TEST 5: Store PDF and Verify
    # ================================================================
    print("\n[TEST 5] Storing PDF and Verifying...")
    try:
        # Store the PDF
        path = await storage.store(session.id, pdf_bytes, metadata={"test": True})
        
        # Verify it exists
        exists = await storage.exists(session.id)
        assert exists, "PDF not found in storage after storing"
        
        # Retrieve and verify content
        retrieved = await storage.retrieve(session.id)
        assert retrieved == pdf_bytes, "Retrieved PDF doesn't match original"
        
        # Get metadata
        metadata = await storage.get_metadata(session.id)
        assert metadata is not None
        assert metadata.size_bytes == len(pdf_bytes)
        
        print(f"   Stored at: {path}")
        print(f"   Size verified: {metadata.size_bytes:,} bytes")
        print(f"   Storage URL: {storage.get_url(session.id)}")
        print("   [PASSED] PDF storage verified")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # ================================================================
    # TEST 6: List Reports
    # ================================================================
    print("\n[TEST 6] Listing Stored Reports...")
    try:
        reports = await storage.list_reports(limit=10)
        
        assert session.id in reports, "Session not found in report list"
        
        print(f"   Found {len(reports)} stored reports")
        print(f"   Session {session.id[:8]}... found in list")
        print("   [PASSED] Report listing works")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        failed += 1
    
    # ================================================================
    # TEST 7: Verify PDF is Valid and Openable
    # ================================================================
    print("\n[TEST 7] Validating PDF Structure...")
    try:
        # Check PDF header
        assert pdf_bytes[:4] == b'%PDF', "Missing PDF header"
        
        # Check for PDF version (should be 1.x)
        header = pdf_bytes[:20].decode('latin-1')
        assert 'PDF-1.' in header, f"Invalid PDF version in header: {header}"
        
        # Check for EOF marker (may have trailing content)
        tail = pdf_bytes[-128:].decode('latin-1', errors='ignore')
        assert '%%EOF' in tail, "Missing EOF marker"
        
        # Check for essential PDF structures
        content_str = pdf_bytes[:2000].decode('latin-1', errors='ignore')
        assert '/Type' in content_str or 'obj' in content_str, "Missing PDF objects"
        
        print(f"   PDF Version: {header.strip()[:8]}")
        print("   EOF marker: Present")
        print("   PDF structure: Valid")
        print("   [PASSED] PDF is valid")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        failed += 1
    
    # ================================================================
    # TEST 8: Delete PDF and Verify
    # ================================================================
    print("\n[TEST 8] Deleting PDF Report...")
    try:
        deleted = await storage.delete(session.id)
        assert deleted, "Delete returned False"
        
        exists_after = await storage.exists(session.id)
        assert not exists_after, "PDF still exists after deletion"
        
        print("   PDF deleted successfully")
        print("   Verified: No longer in storage")
        print("   [PASSED] PDF deletion works")
        passed += 1
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        failed += 1
    
    # ================================================================
    # TEST 9: Empty Session Report
    # ================================================================
    print("\n[TEST 9] Testing Empty Session Report...")
    try:
        # Create a new session without answering any questions
        empty_session, _ = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-empty",
            jd_id="test-jd-empty",
            config=config,
        )
        
        # Try to generate report
        report = await orchestrator.generate_report(empty_session.id)
        
        assert report.session_id == empty_session.id
        assert report.overall_score == 0
        assert report.recommendation == "no_hire"
        assert "no answers" in report.executive_summary.lower() or "insufficient" in report.reasoning.lower()
        
        print("   Empty session handled correctly")
        print(f"   Score: {report.overall_score}, Recommendation: {report.recommendation}")
        print("   [PASSED] Empty session report works")
        passed += 1
        
        # Cleanup
        orchestrator.delete_session(empty_session.id)
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # ================================================================
    # TEST 10: Minimal Session (1 answer) PDF
    # ================================================================
    print("\n[TEST 10] Testing Minimal Session PDF...")
    try:
        # Create a new session
        min_session, _ = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id="test-minimal",
            jd_id="test-jd-minimal",
            config=config,
        )
        
        # Answer just one question
        await orchestrator.submit_answer(
            session_id=min_session.id,
            answer_text="I have 5 years of experience in Python development.",
        )
        
        # Generate PDF
        start_time = time.time()
        pdf_bytes = await orchestrator.export_pdf(min_session.id)
        gen_time = time.time() - start_time
        
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF'
        
        print(f"   Minimal PDF Size: {len(pdf_bytes):,} bytes")
        print(f"   Generation Time: {gen_time:.2f}s")
        print("   [PASSED] Minimal session PDF works")
        passed += 1
        
        # Cleanup
        orchestrator.delete_session(min_session.id)
        
    except Exception as e:
        print(f"   [FAILED] {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # Cleanup main session
    if session:
        orchestrator.delete_session(session.id)
    
    return passed, failed, session.id if session else None


async def main():
    """Main entry point."""
    print("\nStarting PDF Report Generation E2E Test...")
    print("This test requires Ollama with qwen2.5:3b model\n")
    
    try:
        passed, failed, session_id = await run_report_generation_test()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        passed, failed = 0, 1
    
    print("\n" + "=" * 70)
    print("PDF REPORT GENERATION TEST RESULTS")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print("=" * 70)
    
    # Summary of what was tested
    print("\nTests Performed:")
    print("  1. Interview session creation")
    print("  2. Answer submission across stages")
    print("  3. JSON report generation")
    print("  4. PDF generation with <30s performance check")
    print("  5. PDF storage (local provider)")
    print("  6. Report listing")
    print("  7. PDF structure validation")
    print("  8. PDF deletion")
    print("  9. Empty session handling")
    print(" 10. Minimal session (1 answer) PDF")
    print("=" * 70)
    
    if failed > 0:
        print("\nSome tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed! PDF report generation is working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
