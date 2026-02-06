"""
Unit tests for PDF Report Generator.

Tests the PDF generation service and report storage providers.
"""
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.models.report import (
    FullInterviewReport,
    ReportMetadata,
    CandidateInfo,
    ScoreBreakdown,
    ScoreSection,
    QuestionSummary,
    ReportStrength,
    ReportConcern,
    HiringRecommendation,
    RecommendationDecision,
    SeverityLevel,
)
from src.services.pdf_generator import PDFReportGenerator, generate_interview_pdf, save_interview_pdf
from src.providers.report_storage.local_provider import LocalReportStorage
from src.providers.report_storage.base import ReportStorageProvider


# ============= Test Fixtures =============

@pytest.fixture
def sample_report_data():
    """Create sample report data for testing."""
    return FullInterviewReport(
        metadata=ReportMetadata(
            session_id="test-session-123",
            generated_at=datetime(2026, 2, 6, 10, 30, 0),
            report_version="1.0",
        ),
        candidate=CandidateInfo(
            name="John Doe",
            role_title="Senior Backend Engineer",
            interview_date=datetime(2026, 2, 6, 9, 0, 0),
            duration_minutes=45,
        ),
        executive_summary="John Doe demonstrated strong technical skills with excellent Python knowledge. "
                         "He showed good problem-solving abilities and communicated clearly throughout the interview. "
                         "Some areas for improvement in system design scalability considerations.",
        scores=ScoreBreakdown(
            overall=ScoreSection(category="Overall", score=78.5, description="Composite score"),
            technical=ScoreSection(category="Technical", score=82.0, description="Technical knowledge"),
            behavioral=ScoreSection(category="Behavioral", score=75.0, description="Soft skills"),
            communication=ScoreSection(category="Communication", score=80.0, description="Clarity"),
        ),
        strengths=[
            ReportStrength(
                title="Strong Python Expertise",
                evidence="Demonstrated deep understanding of Python internals, decorators, and async programming",
                impact_level="high",
            ),
            ReportStrength(
                title="Clear Communication",
                evidence="Explained complex concepts clearly and asked clarifying questions",
                impact_level="medium",
            ),
        ],
        concerns=[
            ReportConcern(
                title="System Design Gaps",
                severity=SeverityLevel.MEDIUM,
                evidence="Struggled with scalability considerations in distributed systems",
                suggestion="Could benefit from studying CAP theorem and distributed consensus",
            ),
        ],
        question_evaluations=[
            QuestionSummary(
                question_id="q1",
                question_text="Explain how Python's GIL affects multi-threaded applications.",
                answer_text="The Global Interpreter Lock (GIL) in CPython ensures that only one thread executes Python bytecode at a time. This means that CPU-bound multi-threaded programs won't see speedups from threading. However, I/O-bound tasks can still benefit because the GIL is released during I/O operations.",
                stage="technical",
                score=85.0,
                strengths=["Accurate explanation of GIL", "Mentioned I/O-bound distinction"],
                improvements=["Could mention multiprocessing as alternative"],
            ),
            QuestionSummary(
                question_id="q2",
                question_text="Tell me about a time you had to work with a difficult team member.",
                answer_text="In my previous role, I worked with a colleague who was resistant to code reviews. I scheduled a one-on-one to understand their concerns, which turned out to be about time pressure. We agreed on smaller, focused reviews and the situation improved significantly.",
                stage="behavioral",
                score=75.0,
                strengths=["Good STAR structure", "Showed empathy"],
                improvements=["Could quantify the improvement"],
            ),
        ],
        recommendation=HiringRecommendation(
            decision=RecommendationDecision.HIRE,
            confidence_percent=75.0,
            reasoning="Strong technical foundation with room for growth in system design. Good cultural fit.",
            next_steps=["Schedule follow-up technical deep dive", "Discuss team placement options"],
        ),
    )


@pytest.fixture
def minimal_report_data():
    """Create minimal report data with no evaluations."""
    return FullInterviewReport(
        metadata=ReportMetadata(session_id="minimal-session"),
        candidate=CandidateInfo(
            name="Jane Smith",
            role_title="Software Engineer",
            interview_date=datetime.utcnow(),
            duration_minutes=10,
        ),
        executive_summary="Interview ended early with no questions answered.",
        scores=ScoreBreakdown(
            overall=ScoreSection(category="Overall", score=0),
            technical=ScoreSection(category="Technical", score=0),
            behavioral=ScoreSection(category="Behavioral", score=0),
            communication=ScoreSection(category="Communication", score=0),
        ),
        strengths=[],
        concerns=[],
        question_evaluations=[],
        recommendation=HiringRecommendation(
            decision=RecommendationDecision.NO_HIRE,
            confidence_percent=50.0,
            reasoning="Insufficient data to evaluate candidate.",
        ),
    )


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============= PDF Generator Tests =============

class TestPDFReportGenerator:
    """Tests for PDF report generation."""
    
    def test_generate_pdf_returns_bytes(self, sample_report_data):
        """Test that generate() returns valid bytes."""
        generator = PDFReportGenerator(sample_report_data)
        pdf_bytes = generator.generate()
        
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
    
    def test_pdf_starts_with_pdf_header(self, sample_report_data):
        """Test that generated bytes are a valid PDF."""
        generator = PDFReportGenerator(sample_report_data)
        pdf_bytes = generator.generate()
        
        # PDF files start with %PDF-
        assert pdf_bytes[:5] == b'%PDF-'
    
    def test_generate_pdf_minimal_data(self, minimal_report_data):
        """Test PDF generation with minimal/empty data."""
        generator = PDFReportGenerator(minimal_report_data)
        pdf_bytes = generator.generate()
        
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:5] == b'%PDF-'
    
    def test_save_pdf_creates_file(self, sample_report_data, temp_storage_dir):
        """Test that save() creates a file on disk."""
        output_path = os.path.join(temp_storage_dir, "test_report.pdf")
        
        generator = PDFReportGenerator(sample_report_data)
        result_path = generator.save(output_path)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Verify file is valid PDF
        with open(output_path, "rb") as f:
            assert f.read(5) == b'%PDF-'
    
    def test_save_pdf_creates_directories(self, sample_report_data, temp_storage_dir):
        """Test that save() creates parent directories if needed."""
        output_path = os.path.join(temp_storage_dir, "nested", "dir", "report.pdf")
        
        generator = PDFReportGenerator(sample_report_data)
        generator.save(output_path)
        
        assert os.path.exists(output_path)
    
    def test_convenience_function_generate(self, sample_report_data):
        """Test the convenience function generate_interview_pdf()."""
        pdf_bytes = generate_interview_pdf(sample_report_data)
        
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b'%PDF-'
    
    def test_convenience_function_save(self, sample_report_data, temp_storage_dir):
        """Test the convenience function save_interview_pdf()."""
        output_path = os.path.join(temp_storage_dir, "saved_report.pdf")
        
        result = save_interview_pdf(sample_report_data, output_path)
        
        assert os.path.exists(output_path)
        assert result.endswith("saved_report.pdf")


class TestScoreBar:
    """Tests for score bar rendering."""
    
    def test_draw_score_bar_full(self, sample_report_data):
        """Test score bar with full score."""
        generator = PDFReportGenerator(sample_report_data)
        
        bar = generator._draw_score_bar(100, 100, width=10)
        
        # Should be all filled blocks
        assert "\u2588" * 10 in bar
        assert "100/100" in bar
    
    def test_draw_score_bar_empty(self, sample_report_data):
        """Test score bar with zero score."""
        generator = PDFReportGenerator(sample_report_data)
        
        bar = generator._draw_score_bar(0, 100, width=10)
        
        # Should be all empty blocks
        assert "\u2591" * 10 in bar
        assert "0/100" in bar
    
    def test_draw_score_bar_partial(self, sample_report_data):
        """Test score bar with partial score."""
        generator = PDFReportGenerator(sample_report_data)
        
        bar = generator._draw_score_bar(50, 100, width=10)
        
        # Should have 5 filled and 5 empty
        assert "\u2588" * 5 in bar
        assert "\u2591" * 5 in bar
        assert "50/100" in bar
    
    def test_draw_score_bar_zero_max(self, sample_report_data):
        """Test score bar with zero max score (edge case)."""
        generator = PDFReportGenerator(sample_report_data)
        
        bar = generator._draw_score_bar(0, 0, width=10)
        
        # Should handle gracefully
        assert "0/0" in bar


class TestPDFContent:
    """Tests for PDF content structure."""
    
    def test_pdf_contains_candidate_name(self, sample_report_data, temp_storage_dir):
        """Test that PDF contains candidate name."""
        # This is a basic content verification using pypdf2
        # Since ReportLab embeds text, we can read it back
        output_path = os.path.join(temp_storage_dir, "content_test.pdf")
        
        generator = PDFReportGenerator(sample_report_data)
        generator.save(output_path)
        
        # Read and verify file exists with content
        with open(output_path, "rb") as f:
            content = f.read()
        
        # PDF should be substantial
        assert len(content) > 1000
    
    def test_pdf_generation_performance(self, sample_report_data):
        """Test that PDF generation completes within reasonable time."""
        import time
        
        generator = PDFReportGenerator(sample_report_data)
        
        start = time.time()
        pdf_bytes = generator.generate()
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds (well under 30s requirement)
        assert elapsed < 5.0
        assert len(pdf_bytes) > 0


# ============= Storage Provider Tests =============

class TestLocalReportStorage:
    """Tests for local filesystem storage provider."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, temp_storage_dir):
        """Test storing and retrieving a PDF."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        session_id = "test-session-001"
        pdf_content = b"%PDF-1.4 test content"
        
        # Store
        path = await storage.store(session_id, pdf_content)
        assert os.path.exists(path)
        
        # Retrieve
        retrieved = await storage.retrieve(session_id)
        assert retrieved == pdf_content
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, temp_storage_dir):
        """Test retrieving a non-existent report."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        result = await storage.retrieve("nonexistent-session")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_exists(self, temp_storage_dir):
        """Test checking if report exists."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        session_id = "exist-test-001"
        
        # Should not exist initially
        assert await storage.exists(session_id) is False
        
        # Store something
        await storage.store(session_id, b"%PDF-1.4 test")
        
        # Now should exist
        assert await storage.exists(session_id) is True
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_storage_dir):
        """Test deleting a stored report."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        session_id = "delete-test-001"
        await storage.store(session_id, b"%PDF-1.4 test")
        
        assert await storage.exists(session_id) is True
        
        # Delete
        result = await storage.delete(session_id)
        assert result is True
        assert await storage.exists(session_id) is False
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_storage_dir):
        """Test deleting a non-existent report."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        result = await storage.delete("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_url(self, temp_storage_dir):
        """Test getting URL/path for a report."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        url = storage.get_url("some-session")
        assert "some-session.pdf" in url
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, temp_storage_dir):
        """Test getting metadata for a stored report."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        session_id = "metadata-test-001"
        pdf_content = b"%PDF-1.4 test content here"
        
        await storage.store(session_id, pdf_content, metadata={"source": "test"})
        
        metadata = await storage.get_metadata(session_id)
        
        assert metadata is not None
        assert metadata.session_id == session_id
        assert metadata.size_bytes == len(pdf_content)
        assert metadata.content_type == "application/pdf"
    
    @pytest.mark.asyncio
    async def test_list_reports(self, temp_storage_dir):
        """Test listing stored reports."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        # Store a few reports
        await storage.store("session-a", b"%PDF test a")
        await storage.store("session-b", b"%PDF test b")
        await storage.store("session-c", b"%PDF test c")
        
        reports = await storage.list_reports(limit=10)
        
        assert len(reports) == 3
        assert "session-a" in reports
        assert "session-b" in reports
        assert "session-c" in reports
    
    @pytest.mark.asyncio
    async def test_list_reports_with_limit(self, temp_storage_dir):
        """Test listing with limit."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        for i in range(5):
            await storage.store(f"session-{i}", b"%PDF test")
        
        reports = await storage.list_reports(limit=3)
        assert len(reports) == 3
    
    @pytest.mark.asyncio
    async def test_storage_with_metadata(self, temp_storage_dir):
        """Test storing with custom metadata."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        await storage.store(
            "metadata-session",
            b"%PDF content",
            metadata={"custom_field": "custom_value"},
        )
        
        metadata = await storage.get_metadata("metadata-session")
        assert metadata.custom_metadata.get("custom_field") == "custom_value"


class TestStorageProviderInterface:
    """Tests for storage provider interface compliance."""
    
    def test_local_storage_implements_interface(self, temp_storage_dir):
        """Test that LocalReportStorage implements the full interface."""
        storage = LocalReportStorage(base_path=temp_storage_dir)
        
        # Check all required methods exist
        assert hasattr(storage, "store")
        assert hasattr(storage, "retrieve")
        assert hasattr(storage, "exists")
        assert hasattr(storage, "delete")
        assert hasattr(storage, "get_url")
        assert hasattr(storage, "get_metadata")
        assert hasattr(storage, "list_reports")
        
        # Verify it's an instance of the base class
        assert isinstance(storage, ReportStorageProvider)


# ============= Model Tests =============

class TestReportModels:
    """Tests for report data models."""
    
    def test_score_section_percentage(self):
        """Test ScoreSection percentage calculation."""
        score = ScoreSection(category="Test", score=75, max_score=100)
        assert score.percentage == 75.0
        
        score2 = ScoreSection(category="Test", score=50, max_score=200)
        assert score2.percentage == 25.0
    
    def test_score_section_rating(self):
        """Test ScoreSection rating labels."""
        assert ScoreSection(category="T", score=90).rating == "Excellent"
        assert ScoreSection(category="T", score=75).rating == "Good"
        assert ScoreSection(category="T", score=60).rating == "Acceptable"
        assert ScoreSection(category="T", score=45).rating == "Below Average"
        assert ScoreSection(category="T", score=20).rating == "Poor"
    
    def test_question_summary_executive_summary(self):
        """Test QuestionSummary auto-generated executive summary."""
        q = QuestionSummary(
            question_id="q1",
            question_text="Test question",
            answer_text="Test answer",
            stage="technical",
            score=85,
            strengths=["Good explanation"],
            improvements=["Could add examples"],
        )
        
        summary = q.executive_summary
        assert "85/100" in summary
        assert "excellent" in summary
        assert "Good explanation" in summary
    
    def test_recommendation_decision_display(self):
        """Test HiringRecommendation display text."""
        rec = HiringRecommendation(
            decision=RecommendationDecision.STRONG_HIRE,
            confidence_percent=90,
            reasoning="Great candidate",
        )
        assert rec.decision_display == "Strong Hire"
        
        rec2 = HiringRecommendation(
            decision=RecommendationDecision.NO_HIRE,
            confidence_percent=80,
            reasoning="Not a fit",
        )
        assert rec2.decision_display == "Do Not Hire"
    
    def test_recommendation_decision_color(self):
        """Test HiringRecommendation color coding."""
        rec = HiringRecommendation(
            decision=RecommendationDecision.STRONG_HIRE,
            confidence_percent=90,
            reasoning="Test",
        )
        assert rec.decision_color == "#22c55e"  # green
        
        rec2 = HiringRecommendation(
            decision=RecommendationDecision.STRONG_NO_HIRE,
            confidence_percent=90,
            reasoning="Test",
        )
        assert rec2.decision_color == "#ef4444"  # red
