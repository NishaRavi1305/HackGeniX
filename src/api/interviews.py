"""
Interview management API endpoints.

Handles interview sessions, scheduling, and reports.
"""
import logging
from typing import List, Optional
from datetime import datetime
from enum import Enum

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.core.database import mongodb_client

logger = logging.getLogger(__name__)
router = APIRouter()


class InterviewStatus(str, Enum):
    """Interview session status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class InterviewType(str, Enum):
    """Type of interview."""
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    MIXED = "mixed"


class CreateInterviewRequest(BaseModel):
    """Request to create/schedule an interview."""
    candidate_id: str
    job_description_id: str
    interview_type: InterviewType = InterviewType.MIXED
    scheduled_at: Optional[datetime] = None
    duration_minutes: int = 45


class InterviewResponse(BaseModel):
    """Interview session response."""
    id: str
    candidate_id: str
    job_description_id: str
    interview_type: InterviewType
    status: InterviewStatus
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_minutes: int
    created_at: datetime


class InterviewReportSummary(BaseModel):
    """Summary of interview report."""
    interview_id: str
    overall_score: float = Field(ge=0, le=100)
    technical_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    communication_score: Optional[float] = None
    recommendation: str  # strong_hire, hire, no_hire, strong_no_hire
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    summary: str


@router.post("/", response_model=InterviewResponse)
async def create_interview(request: CreateInterviewRequest):
    """
    Create/schedule a new interview session.
    """
    # Validate job description exists
    try:
        jd = await mongodb_client.job_descriptions.find_one(
            {"_id": ObjectId(request.job_description_id)}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job description ID")
    
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")
    
    # Create interview document
    interview = {
        "candidate_id": request.candidate_id,
        "job_description_id": request.job_description_id,
        "interview_type": request.interview_type.value,
        "status": InterviewStatus.SCHEDULED.value,
        "scheduled_at": request.scheduled_at,
        "started_at": None,
        "ended_at": None,
        "duration_minutes": request.duration_minutes,
        "transcript": [],
        "questions_asked": [],
        "responses": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    result = await mongodb_client.interviews.insert_one(interview)
    interview_id = str(result.inserted_id)
    
    logger.info(f"Interview created: {interview_id}")
    
    return InterviewResponse(
        id=interview_id,
        candidate_id=request.candidate_id,
        job_description_id=request.job_description_id,
        interview_type=request.interview_type,
        status=InterviewStatus.SCHEDULED,
        scheduled_at=request.scheduled_at,
        duration_minutes=request.duration_minutes,
        created_at=interview["created_at"],
    )


@router.get("/{interview_id}", response_model=InterviewResponse)
async def get_interview(interview_id: str):
    """
    Get interview details by ID.
    """
    try:
        doc = await mongodb_client.interviews.find_one({"_id": ObjectId(interview_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid interview ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    return InterviewResponse(
        id=str(doc["_id"]),
        candidate_id=doc["candidate_id"],
        job_description_id=doc["job_description_id"],
        interview_type=InterviewType(doc["interview_type"]),
        status=InterviewStatus(doc["status"]),
        scheduled_at=doc.get("scheduled_at"),
        started_at=doc.get("started_at"),
        ended_at=doc.get("ended_at"),
        duration_minutes=doc["duration_minutes"],
        created_at=doc["created_at"],
    )


@router.get("/")
async def list_interviews(
    status: Optional[InterviewStatus] = None,
    skip: int = 0,
    limit: int = 20,
):
    """
    List interviews with optional status filter.
    """
    query = {}
    if status:
        query["status"] = status.value
    
    cursor = mongodb_client.interviews.find(query).skip(skip).limit(limit).sort("created_at", -1)
    interviews = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        interviews.append(doc)
    
    return {"interviews": interviews, "skip": skip, "limit": limit}


@router.post("/{interview_id}/start")
async def start_interview(interview_id: str):
    """
    Start an interview session.
    """
    try:
        doc = await mongodb_client.interviews.find_one({"_id": ObjectId(interview_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid interview ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    if doc["status"] != InterviewStatus.SCHEDULED.value:
        raise HTTPException(status_code=400, detail="Interview cannot be started")
    
    await mongodb_client.interviews.update_one(
        {"_id": ObjectId(interview_id)},
        {
            "$set": {
                "status": InterviewStatus.IN_PROGRESS.value,
                "started_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        }
    )
    
    return {"message": "Interview started", "interview_id": interview_id}


@router.post("/{interview_id}/end")
async def end_interview(interview_id: str):
    """
    End an interview session and trigger report generation.
    """
    try:
        doc = await mongodb_client.interviews.find_one({"_id": ObjectId(interview_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid interview ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    if doc["status"] != InterviewStatus.IN_PROGRESS.value:
        raise HTTPException(status_code=400, detail="Interview is not in progress")
    
    await mongodb_client.interviews.update_one(
        {"_id": ObjectId(interview_id)},
        {
            "$set": {
                "status": InterviewStatus.COMPLETED.value,
                "ended_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        }
    )
    
    # TODO: Trigger report generation in background
    
    return {"message": "Interview ended", "interview_id": interview_id}


@router.get("/{interview_id}/report", response_model=InterviewReportSummary)
async def get_interview_report(interview_id: str):
    """
    Get the generated report for a completed interview.
    """
    try:
        doc = await mongodb_client.interviews.find_one({"_id": ObjectId(interview_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid interview ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    if doc["status"] != InterviewStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Interview is not completed")
    
    # Check for existing report
    report = await mongodb_client.reports.find_one({"interview_id": interview_id})
    
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Report not yet generated. Please wait for processing."
        )
    
    return InterviewReportSummary(
        interview_id=interview_id,
        overall_score=report.get("overall_score", 0),
        technical_score=report.get("technical_score"),
        behavioral_score=report.get("behavioral_score"),
        communication_score=report.get("communication_score"),
        recommendation=report.get("recommendation", "pending"),
        strengths=report.get("strengths", []),
        weaknesses=report.get("weaknesses", []),
        summary=report.get("summary", "Report generation in progress"),
    )
