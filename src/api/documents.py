"""
Document management API endpoints.

Handles resume and job description uploads, parsing, and matching.
"""
import logging
from typing import List
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.core.database import get_db, mongodb_client
from src.core.storage import get_storage, StorageClient
from src.core.auth import get_current_user, require_role, require_permission
from src.core.permissions import Permissions
from src.models.auth import AuthenticatedUser, UserRole
from src.models.documents import (
    ResumeUploadResponse,
    JobDescriptionCreateRequest,
    JobDescriptionResponse,
    MatchResult,
    ResumeDocument,
    JobDescriptionDocument,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Allowed file types for resume upload
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/resumes", response_model=ResumeUploadResponse)
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    storage: StorageClient = Depends(get_storage),
    user: AuthenticatedUser = Depends(require_permission(Permissions.UPLOAD_DOCUMENT)),
):
    """
    Upload a resume file for parsing and analysis.
    
    Supported formats: PDF, DOC, DOCX
    Max file size: 10 MB
    """
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: PDF, DOC, DOCX"
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: 10 MB"
        )
    
    # Upload to storage
    storage_key = await storage.upload_bytes(
        data=content,
        filename=file.filename,
        category="resumes",
        content_type=file.content_type,
    )
    
    # Create document record
    doc = ResumeDocument(
        filename=file.filename,
        storage_key=storage_key,
        content_type=file.content_type,
        file_size=file_size,
        status="pending",
    )
    
    # Insert into database
    result = await mongodb_client.resumes.insert_one(
        doc.model_dump(by_alias=True, exclude={"id"})
    )
    doc_id = str(result.inserted_id)
    
    # Queue background parsing task
    # background_tasks.add_task(parse_resume_task, doc_id)
    
    logger.info(f"Resume uploaded: {file.filename} -> {doc_id}")
    
    return ResumeUploadResponse(
        id=doc_id,
        filename=file.filename,
        status="pending",
        message="Resume uploaded successfully. Parsing in progress.",
    )


@router.get("/resumes/{resume_id}")
async def get_resume(
    resume_id: str,
    user: AuthenticatedUser = Depends(require_permission(Permissions.VIEW_DOCUMENT)),
):
    """
    Get resume details by ID.
    """
    try:
        doc = await mongodb_client.resumes.find_one({"_id": ObjectId(resume_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resume ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/resumes")
async def list_resumes(
    skip: int = 0,
    limit: int = 20,
    user: AuthenticatedUser = Depends(require_permission(Permissions.VIEW_DOCUMENT)),
):
    """
    List all resumes with pagination.
    """
    cursor = mongodb_client.resumes.find().skip(skip).limit(limit).sort("created_at", -1)
    resumes = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        resumes.append(doc)
    return {"resumes": resumes, "skip": skip, "limit": limit}


@router.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    storage: StorageClient = Depends(get_storage),
    user: AuthenticatedUser = Depends(require_permission(Permissions.DELETE_DOCUMENT)),
):
    """
    Delete a resume by ID.
    """
    try:
        doc = await mongodb_client.resumes.find_one({"_id": ObjectId(resume_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resume ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Delete from storage
    await storage.delete_file(doc["storage_key"])
    
    # Delete from database
    await mongodb_client.resumes.delete_one({"_id": ObjectId(resume_id)})
    
    return {"message": "Resume deleted successfully"}


# Job Description Endpoints

@router.post("/job-descriptions", response_model=JobDescriptionResponse)
async def create_job_description(
    request: JobDescriptionCreateRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(require_permission(Permissions.UPLOAD_DOCUMENT)),
):
    """
    Create a new job description for matching.
    """
    doc = JobDescriptionDocument(
        title=request.title,
        company=request.company,
        status="pending",
    )
    
    # Store raw text for parsing
    doc_dict = doc.model_dump(by_alias=True, exclude={"id"})
    doc_dict["raw_text"] = request.description
    
    result = await mongodb_client.job_descriptions.insert_one(doc_dict)
    doc_id = str(result.inserted_id)
    
    # Queue background parsing task
    # background_tasks.add_task(parse_job_description_task, doc_id)
    
    logger.info(f"Job description created: {request.title} -> {doc_id}")
    
    return JobDescriptionResponse(
        id=doc_id,
        title=request.title,
        company=request.company,
        status="pending",
    )


@router.get("/job-descriptions/{jd_id}")
async def get_job_description(
    jd_id: str,
    user: AuthenticatedUser = Depends(require_permission(Permissions.VIEW_DOCUMENT)),
):
    """
    Get job description details by ID.
    """
    try:
        doc = await mongodb_client.job_descriptions.find_one({"_id": ObjectId(jd_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job description ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Job description not found")
    
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/job-descriptions")
async def list_job_descriptions(
    skip: int = 0,
    limit: int = 20,
    user: AuthenticatedUser = Depends(require_permission(Permissions.VIEW_DOCUMENT)),
):
    """
    List all job descriptions with pagination.
    """
    cursor = mongodb_client.job_descriptions.find().skip(skip).limit(limit).sort("created_at", -1)
    jds = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        jds.append(doc)
    return {"job_descriptions": jds, "skip": skip, "limit": limit}


# Matching Endpoints

@router.post("/match", response_model=MatchResult)
async def match_resume_to_job(
    resume_id: str,
    job_description_id: str,
    user: AuthenticatedUser = Depends(require_permission(Permissions.RUN_ANALYSIS)),
):
    """
    Match a resume against a job description.
    
    Returns match scores and analysis.
    """
    # Validate resume exists
    try:
        resume = await mongodb_client.resumes.find_one({"_id": ObjectId(resume_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resume ID")
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Validate job description exists
    try:
        jd = await mongodb_client.job_descriptions.find_one({"_id": ObjectId(job_description_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job description ID")
    
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")
    
    # Check if both documents are parsed
    if resume.get("status") != "parsed" or jd.get("status") != "parsed":
        raise HTTPException(
            status_code=400,
            detail="Both resume and job description must be parsed before matching"
        )
    
    # TODO: Call semantic matcher service
    # For now, return placeholder
    return MatchResult(
        resume_id=resume_id,
        job_description_id=job_description_id,
        overall_score=0,
        skill_match_score=0,
        experience_match_score=0,
        semantic_similarity_score=0,
        matched_skills=[],
        missing_skills=[],
        recommendations=["Documents need to be processed by the matching service"],
    )
