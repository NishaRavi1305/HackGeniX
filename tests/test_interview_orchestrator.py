"""
Unit tests for Interview Orchestrator Service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.models.interview import (
    InterviewSession,
    InterviewConfig,
    InterviewStatus,
    InterviewStage,
    InterviewMode,
    InterviewQuestion,
    AnswerRecord,
    StageProgress,
    QuestionStatus,
)
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo
from src.services.interview_orchestrator import InterviewOrchestrator, STAGE_ORDER
from src.services.question_generator import GeneratedQuestion
from src.services.answer_evaluator import AnswerEvaluation, EvaluationScores, AnswerStrength
from src.services.prompts import InterviewStage as PromptStage, QuestionDifficulty


# Fixtures

@pytest.fixture
def mock_resume():
    """Create a mock parsed resume."""
    return ParsedResume(
        contact=ContactInfo(name="John Doe", email="john@example.com"),
        summary="Experienced software engineer with 5+ years in backend development.",
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
        experience=[
            {
                "company": "Tech Corp",
                "title": "Senior Software Engineer",
                "description": "Led backend development team.",
            }
        ],
        education=[{"institution": "State University", "degree": "B.S. Computer Science"}],
        raw_text="John Doe - Senior Software Engineer",
    )


@pytest.fixture
def mock_jd():
    """Create a mock parsed job description."""
    return ParsedJobDescription(
        title="Senior Backend Engineer",
        company="HackGeniX Tech",
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_skills=["Kubernetes", "AWS"],
        responsibilities=["Design scalable systems", "Write clean code"],
        experience_years_min=5,
        raw_text="Senior Backend Engineer position.",
    )


@pytest.fixture
def mock_question_generator():
    """Create a mock question generator."""
    generator = MagicMock()
    
    async def mock_generate(request, resume, jd, previous_questions=None):
        # Return mock questions based on stage
        stage = request.stage
        return [
            GeneratedQuestion(
                question=f"Test {stage.value} question {i+1}",
                stage=stage,
                difficulty=QuestionDifficulty.MEDIUM,
                category="test",
                purpose="Testing",
                expected_answer_points=["Point 1", "Point 2"],
            )
            for i in range(request.num_questions)
        ]
    
    generator.generate_questions = AsyncMock(side_effect=mock_generate)
    return generator


@pytest.fixture
def mock_answer_evaluator():
    """Create a mock answer evaluator."""
    evaluator = MagicMock()
    
    async def mock_evaluate(question, answer, expected_points, stage, validate=True):
        return AnswerEvaluation(
            question=question,
            answer=answer,
            stage=stage,
            scores=EvaluationScores(
                technical_accuracy=80,
                completeness=75,
                clarity=85,
                depth=70,
                overall=77.5,
            ),
            strengths=["Good explanation", "Clear communication"],
            improvements=["Could add more detail"],
            recommendation=AnswerStrength.ACCEPTABLE,
            is_validated=True,
        )
    
    async def mock_evaluate_behavioral(question, answer, competency, red_flags, green_flags, validate=True):
        return AnswerEvaluation(
            question=question,
            answer=answer,
            stage=PromptStage.BEHAVIORAL,
            scores=EvaluationScores(overall=80),
            strengths=["Good STAR structure"],
            recommendation=AnswerStrength.STRONG,
            is_validated=True,
        )
    
    async def mock_generate_report(candidate_name, role_title, duration_minutes, evaluations):
        from src.services.answer_evaluator import InterviewReport, Recommendation
        return InterviewReport(
            candidate_name=candidate_name,
            role_title=role_title,
            duration_minutes=duration_minutes,
            evaluations=evaluations,
            technical_score=75,
            behavioral_score=80,
            communication_score=85,
            problem_solving_score=70,
            overall_score=77,
            executive_summary="Solid candidate with good technical skills.",
            recommendation=Recommendation.HIRE,
            confidence=75,
            reasoning="Meets requirements",
        )
    
    evaluator.evaluate_answer = AsyncMock(side_effect=mock_evaluate)
    evaluator.evaluate_behavioral_answer = AsyncMock(side_effect=mock_evaluate_behavioral)
    evaluator.generate_interview_report = AsyncMock(side_effect=mock_generate_report)
    
    return evaluator


@pytest.fixture
def orchestrator(mock_question_generator, mock_answer_evaluator):
    """Create an orchestrator with mocked dependencies."""
    return InterviewOrchestrator(
        question_generator=mock_question_generator,
        answer_evaluator=mock_answer_evaluator,
    )


# Tests for InterviewSession model

class TestInterviewSessionModel:
    """Tests for InterviewSession model."""
    
    def test_create_session(self):
        """Test creating an interview session."""
        session = InterviewSession(
            id="test-123",
            resume_id="resume-1",
            job_description_id="jd-1",
        )
        
        assert session.id == "test-123"
        assert session.status == InterviewStatus.CREATED
        assert session.current_stage == InterviewStage.SCREENING
        assert len(session.questions) == 0
        assert len(session.answers) == 0
    
    def test_session_overall_score(self):
        """Test overall score calculation."""
        session = InterviewSession(
            id="test-123",
            resume_id="resume-1",
            job_description_id="jd-1",
        )
        
        # Add some answers with scores
        session.answers = [
            AnswerRecord(
                question_id="q1",
                question_text="Q1",
                stage=InterviewStage.TECHNICAL,
                answer_text="Answer 1",
                scores={"overall": 80},
            ),
            AnswerRecord(
                question_id="q2",
                question_text="Q2",
                stage=InterviewStage.TECHNICAL,
                answer_text="Answer 2",
                scores={"overall": 70},
            ),
        ]
        
        assert session.overall_score == 75.0
    
    def test_current_question(self):
        """Test getting current question."""
        session = InterviewSession(
            id="test-123",
            resume_id="resume-1",
            job_description_id="jd-1",
        )
        
        # Add questions
        session.questions = [
            InterviewQuestion(id="q1", question_text="Q1", stage=InterviewStage.SCREENING, status=QuestionStatus.ANSWERED),
            InterviewQuestion(id="q2", question_text="Q2", stage=InterviewStage.SCREENING, status=QuestionStatus.PENDING),
            InterviewQuestion(id="q3", question_text="Q3", stage=InterviewStage.TECHNICAL, status=QuestionStatus.PENDING),
        ]
        
        current = session.current_question
        assert current is not None
        assert current.id == "q2"


# Tests for InterviewOrchestrator

class TestInterviewOrchestrator:
    """Tests for InterviewOrchestrator service."""
    
    @pytest.mark.asyncio
    async def test_start_interview(self, orchestrator, mock_resume, mock_jd):
        """Test starting a new interview."""
        session, response = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
        )
        
        assert session.id is not None
        assert session.status == InterviewStatus.IN_PROGRESS
        assert session.candidate_name == "John Doe"
        assert session.role_title == "Senior Backend Engineer"
        assert len(session.questions) > 0
        assert session.started_at is not None
        
        # Check response
        assert response.session_id == session.id
        assert response.first_question is not None
        assert response.total_questions == len(session.questions)
    
    @pytest.mark.asyncio
    async def test_start_interview_with_config(self, orchestrator, mock_resume, mock_jd):
        """Test starting interview with custom config."""
        config = InterviewConfig(
            screening_questions=2,
            technical_questions=3,
            behavioral_questions=2,
            system_design_questions=0,
        )
        
        session, response = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
            config=config,
        )
        
        # Count questions per stage
        screening = session.get_stage_questions(InterviewStage.SCREENING)
        technical = session.get_stage_questions(InterviewStage.TECHNICAL)
        behavioral = session.get_stage_questions(InterviewStage.BEHAVIORAL)
        
        assert len(screening) == 2
        assert len(technical) == 3
        assert len(behavioral) == 2
    
    @pytest.mark.asyncio
    async def test_submit_answer(self, orchestrator, mock_resume, mock_jd):
        """Test submitting an answer."""
        # Start interview first
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
        )
        
        # Submit answer
        response = await orchestrator.submit_answer(
            session_id=session.id,
            answer_text="This is my answer to the first question.",
        )
        
        assert response.session_id == session.id
        assert response.evaluation is not None
        assert response.questions_answered == 1
        assert response.progress_percent > 0
        
        # Check session was updated
        updated_session = orchestrator.get_session(session.id)
        assert len(updated_session.answers) == 1
    
    @pytest.mark.asyncio
    async def test_submit_answer_stage_transition(self, orchestrator, mock_resume, mock_jd):
        """Test that stage transitions happen correctly."""
        config = InterviewConfig(
            screening_questions=1,  # Only 1 screening question
            technical_questions=2,
            behavioral_questions=1,
            system_design_questions=0,
        )
        
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
            config=config,
        )
        
        # Answer screening question
        response = await orchestrator.submit_answer(
            session_id=session.id,
            answer_text="Screening answer",
        )
        
        # Should transition to technical
        assert response.stage_changed is True
        assert response.current_stage == "technical"
    
    @pytest.mark.asyncio
    async def test_submit_answer_interview_complete(self, orchestrator, mock_resume, mock_jd):
        """Test that interview completes after all questions."""
        config = InterviewConfig(
            screening_questions=1,
            technical_questions=1,
            behavioral_questions=1,
            system_design_questions=0,
        )
        
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
            config=config,
        )
        
        # Answer all questions
        total_questions = len(session.questions)
        for i in range(total_questions):
            response = await orchestrator.submit_answer(
                session_id=session.id,
                answer_text=f"Answer {i+1}",
            )
        
        # Last response should indicate completion
        assert response.interview_complete is True
        
        updated = orchestrator.get_session(session.id)
        assert updated.status == InterviewStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_progress(self, orchestrator, mock_resume, mock_jd):
        """Test getting interview progress."""
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
        )
        
        # Submit one answer
        await orchestrator.submit_answer(
            session_id=session.id,
            answer_text="My answer",
        )
        
        progress = orchestrator.get_progress(session.id)
        
        assert progress.session_id == session.id
        assert progress.questions_answered == 1
        assert progress.total_questions == len(session.questions)
        assert progress.progress_percent > 0
    
    @pytest.mark.asyncio
    async def test_end_interview_early(self, orchestrator, mock_resume, mock_jd):
        """Test ending interview early."""
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
        )
        
        # Answer one question
        await orchestrator.submit_answer(
            session_id=session.id,
            answer_text="My answer",
        )
        
        # End early
        report = await orchestrator.end_interview(
            session_id=session.id,
            reason="Time constraint",
        )
        
        assert report.session_id == session.id
        assert report.candidate_name == "John Doe"
        assert report.overall_score >= 0
        
        updated = orchestrator.get_session(session.id)
        assert updated.status == InterviewStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_generate_report(self, orchestrator, mock_resume, mock_jd):
        """Test generating interview report."""
        config = InterviewConfig(
            screening_questions=1,
            technical_questions=1,
            behavioral_questions=0,
            system_design_questions=0,
        )
        
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
            config=config,
        )
        
        # Answer all questions
        for _ in range(len(session.questions)):
            await orchestrator.submit_answer(
                session_id=session.id,
                answer_text="My detailed answer",
            )
        
        report = await orchestrator.generate_report(session.id)
        
        assert report.session_id == session.id
        assert report.candidate_name == "John Doe"
        assert report.role_title == "Senior Backend Engineer"
        assert report.overall_score > 0
        assert report.recommendation in ["strong_hire", "hire", "no_hire", "strong_no_hire"]
    
    def test_pause_resume_interview(self, orchestrator):
        """Test pausing and resuming interview."""
        # Create a manual session for this test
        session = InterviewSession(
            id="test-pause",
            resume_id="resume-1",
            job_description_id="jd-1",
            status=InterviewStatus.IN_PROGRESS,
        )
        orchestrator._sessions["test-pause"] = session
        
        # Pause
        paused = orchestrator.pause_interview("test-pause")
        assert paused.status == InterviewStatus.PAUSED
        
        # Resume
        resumed = orchestrator.resume_interview("test-pause")
        assert resumed.status == InterviewStatus.IN_PROGRESS
    
    def test_list_sessions(self, orchestrator):
        """Test listing sessions."""
        # Create some sessions
        for i in range(3):
            session = InterviewSession(
                id=f"session-{i}",
                resume_id="resume-1",
                job_description_id="jd-1",
                status=InterviewStatus.IN_PROGRESS if i < 2 else InterviewStatus.COMPLETED,
            )
            orchestrator._sessions[session.id] = session
        
        # List all
        all_sessions = orchestrator.list_sessions()
        assert len(all_sessions) == 3
        
        # List by status
        in_progress = orchestrator.list_sessions(status=InterviewStatus.IN_PROGRESS)
        assert len(in_progress) == 2
    
    def test_delete_session(self, orchestrator):
        """Test deleting a session."""
        session = InterviewSession(
            id="to-delete",
            resume_id="resume-1",
            job_description_id="jd-1",
        )
        orchestrator._sessions["to-delete"] = session
        
        assert orchestrator.delete_session("to-delete") is True
        assert orchestrator.get_session("to-delete") is None
        
        # Try to delete non-existent
        assert orchestrator.delete_session("non-existent") is False
    
    @pytest.mark.asyncio
    async def test_submit_answer_not_found(self, orchestrator):
        """Test submitting answer for non-existent session."""
        with pytest.raises(ValueError, match="Session not found"):
            await orchestrator.submit_answer(
                session_id="non-existent",
                answer_text="Answer",
            )
    
    @pytest.mark.asyncio
    async def test_submit_answer_no_text(self, orchestrator, mock_resume, mock_jd):
        """Test submitting empty answer."""
        session, _ = await orchestrator.start_interview(
            resume=mock_resume,
            jd=mock_jd,
            resume_id="resume-1",
            jd_id="jd-1",
        )
        
        with pytest.raises(ValueError, match="No answer provided"):
            await orchestrator.submit_answer(
                session_id=session.id,
                answer_text=None,
            )


class TestInterviewConfig:
    """Tests for InterviewConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = InterviewConfig()
        
        assert config.mode == InterviewMode.TEXT
        assert config.screening_questions == 3
        assert config.technical_questions == 5
        assert config.behavioral_questions == 3
        assert config.system_design_questions == 1
        assert config.adaptive_difficulty is True
        assert config.enable_follow_ups is True
    
    def test_voice_config(self):
        """Test voice mode configuration."""
        config = InterviewConfig(
            mode=InterviewMode.VOICE,
            tts_voice="Microsoft David",
            stt_model="small",
        )
        
        assert config.mode == InterviewMode.VOICE
        assert config.tts_voice == "Microsoft David"
        assert config.stt_model == "small"


class TestStageProgress:
    """Tests for StageProgress model."""
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = StageProgress(
            stage=InterviewStage.TECHNICAL,
            total_questions=5,
            answered_questions=2,
        )
        
        assert progress.progress_percent == 40.0
        assert progress.is_complete is False
    
    def test_complete_stage(self):
        """Test complete stage detection."""
        progress = StageProgress(
            stage=InterviewStage.SCREENING,
            total_questions=3,
            answered_questions=3,
        )
        
        assert progress.is_complete is True
        assert progress.progress_percent == 100.0


class TestStageOrder:
    """Tests for stage ordering."""
    
    def test_stage_order(self):
        """Test that stages are in correct order."""
        assert STAGE_ORDER == [
            InterviewStage.SCREENING,
            InterviewStage.TECHNICAL,
            InterviewStage.BEHAVIORAL,
            InterviewStage.SYSTEM_DESIGN,
            InterviewStage.WRAP_UP,
        ]
