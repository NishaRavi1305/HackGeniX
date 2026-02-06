"""
Interview Session API endpoints.

Provides REST API for managing complete interview sessions.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Query

from src.models.interview import (
    InterviewSession,
    InterviewConfig,
    InterviewStatus,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    InterviewProgressResponse,
    EndInterviewRequest,
    InterviewReportResponse,
)
from src.models.documents import ParsedResume, ParsedJobDescription
from src.services.interview_orchestrator import get_interview_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


# Mock document storage (in production, this would be a database)
# For now, we'll store parsed documents here for testing
_mock_resumes: dict = {}
_mock_jds: dict = {}


def _get_mock_resume() -> ParsedResume:
    """Get a mock resume for testing."""
    return ParsedResume(
        contact={"name": "John Doe", "email": "john@example.com"},
        summary="Experienced software engineer with 5+ years in backend development.",
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS", "Redis", "GraphQL"],
        experience=[
            {
                "company": "Tech Corp",
                "title": "Senior Software Engineer",
                "description": "Led backend development team, designed microservices architecture.",
            },
            {
                "company": "StartupXYZ",
                "title": "Software Engineer",
                "description": "Built REST APIs and data pipelines.",
            },
        ],
        education=[
            {
                "institution": "State University",
                "degree": "B.S. Computer Science",
            }
        ],
        raw_text="John Doe - Senior Software Engineer with expertise in Python and cloud technologies.",
    )


def _get_mock_jd() -> ParsedJobDescription:
    """Get a mock job description for testing."""
    return ParsedJobDescription(
        title="Senior Backend Engineer",
        company="HackGeniX Tech",
        required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs"],
        preferred_skills=["Kubernetes", "AWS", "Redis", "GraphQL", "CI/CD"],
        responsibilities=[
            "Design and implement scalable backend services",
            "Write clean, maintainable, and well-tested code",
            "Collaborate with cross-functional teams",
            "Mentor junior developers",
        ],
        qualifications=[
            "5+ years of software engineering experience",
            "Strong Python skills",
            "Experience with cloud platforms",
        ],
        experience_years_min=5,
        raw_text="Senior Backend Engineer position at HackGeniX Tech.",
    )


@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview session.
    
    Creates an interview session with questions generated based on the
    candidate's resume and the job description.
    """
    orchestrator = get_interview_orchestrator()
    
    try:
        # In production, fetch from database
        # For now, use mock data or stored data
        resume = _mock_resumes.get(request.resume_id) or _get_mock_resume()
        jd = _mock_jds.get(request.job_description_id) or _get_mock_jd()
        
        session, response = await orchestrator.start_interview(
            resume=resume,
            jd=jd,
            resume_id=request.resume_id,
            jd_id=request.job_description_id,
            config=request.config,
        )
        
        logger.info(f"Interview started: session={session.id}, questions={len(session.questions)}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to start interview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview: {str(e)}",
        )


@router.post("/answer", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """
    Submit an answer for the current question.
    
    Accepts text answers or audio (for voice mode).
    Returns evaluation and next question.
    """
    orchestrator = get_interview_orchestrator()
    
    try:
        response = await orchestrator.submit_answer(
            session_id=request.session_id,
            answer_text=request.answer_text,
            answer_audio_base64=request.answer_audio_base64,
        )
        
        logger.info(
            f"Answer submitted: session={request.session_id}, "
            f"score={response.evaluation.get('scores', {}).get('overall', 0) if response.evaluation else 0}"
        )
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to submit answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit answer: {str(e)}",
        )


@router.get("/{session_id}/progress", response_model=InterviewProgressResponse)
async def get_progress(session_id: str):
    """Get current interview progress and state."""
    orchestrator = get_interview_orchestrator()
    
    try:
        return orchestrator.get_progress(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/{session_id}", response_model=dict)
async def get_session(session_id: str):
    """Get full interview session details."""
    orchestrator = get_interview_orchestrator()
    
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return session.model_dump()


@router.post("/{session_id}/end", response_model=InterviewReportResponse)
async def end_interview(session_id: str, request: Optional[EndInterviewRequest] = None):
    """
    End an interview and generate the final report.
    
    Can be called to end early or after all questions are answered.
    """
    orchestrator = get_interview_orchestrator()
    
    try:
        reason = request.reason if request else None
        return await orchestrator.end_interview(session_id, reason)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to end interview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end interview: {str(e)}",
        )


@router.get("/{session_id}/report", response_model=InterviewReportResponse)
async def get_report(session_id: str):
    """
    Get the interview report.
    
    Generates a comprehensive report with scores and recommendations.
    """
    orchestrator = get_interview_orchestrator()
    
    try:
        return await orchestrator.generate_report(session_id)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}",
        )


@router.post("/{session_id}/pause")
async def pause_interview(session_id: str):
    """Pause an ongoing interview."""
    orchestrator = get_interview_orchestrator()
    
    try:
        session = orchestrator.pause_interview(session_id)
        return {
            "session_id": session_id,
            "status": session.status.value,
            "message": "Interview paused",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{session_id}/resume")
async def resume_interview(session_id: str):
    """Resume a paused interview."""
    orchestrator = get_interview_orchestrator()
    
    try:
        session = orchestrator.resume_interview(session_id)
        return {
            "session_id": session_id,
            "status": session.status.value,
            "current_question": session.current_question,
            "message": "Interview resumed",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/", response_model=List[dict])
async def list_sessions(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
):
    """List interview sessions."""
    orchestrator = get_interview_orchestrator()
    
    status_filter = None
    if status:
        try:
            status_filter = InterviewStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}",
            )
    
    sessions = orchestrator.list_sessions(status=status_filter, limit=limit)
    
    return [
        {
            "id": s.id,
            "candidate_name": s.candidate_name,
            "role_title": s.role_title,
            "status": s.status.value if hasattr(s.status, 'value') else s.status,
            "current_stage": s.current_stage.value if hasattr(s.current_stage, 'value') else s.current_stage,
            "questions_answered": len(s.answers),
            "total_questions": len(s.questions),
            "overall_score": s.overall_score,
            "duration_minutes": s.duration_minutes,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete an interview session."""
    orchestrator = get_interview_orchestrator()
    
    if orchestrator.delete_session(session_id):
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


# Endpoints for managing mock documents (for testing)

@router.post("/mock/resume")
async def add_mock_resume(resume_id: str, resume: ParsedResume):
    """Add a mock resume for testing."""
    _mock_resumes[resume_id] = resume
    return {"message": f"Resume {resume_id} stored", "id": resume_id}


@router.post("/mock/jd")
async def add_mock_jd(jd_id: str, jd: ParsedJobDescription):
    """Add a mock job description for testing."""
    _mock_jds[jd_id] = jd
    return {"message": f"Job description {jd_id} stored", "id": jd_id}
