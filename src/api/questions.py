"""
Question Generation and Evaluation API endpoints.

Handles LLM-powered question generation and answer evaluation.
"""
import logging
from typing import List, Optional
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.core.database import mongodb_client
from src.services.question_generator import (
    QuestionGenerator,
    get_question_generator,
    QuestionGenerationRequest,
    GeneratedQuestion,
)
from src.services.answer_evaluator import (
    AnswerEvaluator,
    get_answer_evaluator,
    AnswerEvaluation,
)
from src.services.prompts import InterviewStage, QuestionDifficulty
from src.models.documents import ParsedResume, ParsedJobDescription

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models

class GenerateQuestionsRequest(BaseModel):
    """Request to generate interview questions."""
    job_description_id: str
    resume_id: Optional[str] = None
    stage: str = "technical"  # screening, technical, behavioral, system_design
    num_questions: int = Field(default=5, ge=1, le=10)
    difficulty: str = "medium"  # easy, medium, hard
    focus_areas: List[str] = Field(default_factory=list)


class GeneratedQuestionResponse(BaseModel):
    """Response model for a generated question."""
    question: str
    stage: str
    difficulty: str
    category: Optional[str] = None
    purpose: str = ""
    expected_answer_points: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    duration_seconds: int = 120


class GenerateQuestionsResponse(BaseModel):
    """Response containing generated questions."""
    questions: List[GeneratedQuestionResponse]
    job_description_id: str
    resume_id: Optional[str] = None
    stage: str
    generated_at: datetime


class EvaluateAnswerRequest(BaseModel):
    """Request to evaluate a candidate's answer."""
    question: str
    answer: str
    expected_points: List[str] = Field(default_factory=list)
    stage: str = "technical"
    validate: bool = True  # Run hallucination check


class EvaluationScoresResponse(BaseModel):
    """Evaluation scores response."""
    technical_accuracy: float = 0
    completeness: float = 0
    clarity: float = 0
    depth: float = 0
    overall: float = 0


class EvaluateAnswerResponse(BaseModel):
    """Response from answer evaluation."""
    question: str
    answer: str
    stage: str
    scores: EvaluationScoresResponse
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    follow_up_question: Optional[str] = None
    recommendation: str
    notes: str = ""
    is_validated: bool = False
    validation_issues: List[str] = Field(default_factory=list)


class EvaluateBehavioralRequest(BaseModel):
    """Request to evaluate a behavioral answer."""
    question: str
    answer: str
    competency: str = "problem-solving"
    red_flags: List[str] = Field(default_factory=list)
    green_flags: List[str] = Field(default_factory=list)
    validate: bool = True


class GenerateFollowUpRequest(BaseModel):
    """Request to generate a follow-up question."""
    original_question: str
    candidate_answer: str
    evaluation_summary: str


class FollowUpResponse(BaseModel):
    """Response with follow-up question."""
    follow_up_question: Optional[str] = None


# Endpoints

@router.post("/questions/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    """
    Generate interview questions based on job description and resume.
    
    Uses LLM to create contextual, relevant questions for the specified
    interview stage.
    """
    # Validate job description exists
    try:
        jd_doc = await mongodb_client.job_descriptions.find_one(
            {"_id": ObjectId(request.job_description_id)}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job description ID")
    
    if not jd_doc:
        raise HTTPException(status_code=404, detail="Job description not found")
    
    # Get resume if provided
    resume_doc = None
    if request.resume_id:
        try:
            resume_doc = await mongodb_client.resumes.find_one(
                {"_id": ObjectId(request.resume_id)}
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid resume ID")
        
        if not resume_doc:
            raise HTTPException(status_code=404, detail="Resume not found")
    
    # Build parsed objects
    jd = ParsedJobDescription(
        title=jd_doc.get("title", ""),
        company=jd_doc.get("company"),
        required_skills=jd_doc.get("parsed_data", {}).get("required_skills", []),
        preferred_skills=jd_doc.get("parsed_data", {}).get("preferred_skills", []),
        responsibilities=jd_doc.get("parsed_data", {}).get("responsibilities", []),
        experience_years_min=jd_doc.get("parsed_data", {}).get("experience_years_min"),
        raw_text=jd_doc.get("raw_text", ""),
    )
    
    resume = ParsedResume(
        summary=resume_doc.get("parsed_data", {}).get("summary") if resume_doc else None,
        skills=resume_doc.get("parsed_data", {}).get("skills", []) if resume_doc else [],
        raw_text=resume_doc.get("parsed_data", {}).get("raw_text", "") if resume_doc else "",
    )
    
    # Map stage string to enum
    try:
        stage = InterviewStage(request.stage)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid stage. Must be one of: {[s.value for s in InterviewStage]}"
        )
    
    try:
        difficulty = QuestionDifficulty(request.difficulty)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty. Must be one of: {[d.value for d in QuestionDifficulty]}"
        )
    
    # Generate questions
    generator = get_question_generator()
    gen_request = QuestionGenerationRequest(
        stage=stage,
        num_questions=request.num_questions,
        difficulty=difficulty,
        focus_areas=request.focus_areas,
    )
    
    try:
        questions = await generator.generate_questions(gen_request, resume, jd)
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")
    
    # Convert to response format
    response_questions = [
        GeneratedQuestionResponse(
            question=q.question,
            stage=q.stage.value,
            difficulty=q.difficulty.value,
            category=q.category,
            purpose=q.purpose,
            expected_answer_points=q.expected_answer_points,
            follow_up_questions=q.follow_up_questions,
            duration_seconds=q.duration_seconds,
        )
        for q in questions
    ]
    
    logger.info(f"Generated {len(questions)} questions for JD {request.job_description_id}")
    
    return GenerateQuestionsResponse(
        questions=response_questions,
        job_description_id=request.job_description_id,
        resume_id=request.resume_id,
        stage=request.stage,
        generated_at=datetime.utcnow(),
    )


@router.post("/answers/evaluate", response_model=EvaluateAnswerResponse)
async def evaluate_answer(request: EvaluateAnswerRequest):
    """
    Evaluate a candidate's answer to an interview question.
    
    Uses LLM-as-judge to assess the answer on multiple criteria
    including technical accuracy, completeness, clarity, and depth.
    """
    # Map stage
    try:
        stage = InterviewStage(request.stage)
    except ValueError:
        stage = InterviewStage.TECHNICAL
    
    evaluator = get_answer_evaluator()
    
    try:
        evaluation = await evaluator.evaluate_answer(
            question=request.question,
            answer=request.answer,
            expected_points=request.expected_points,
            stage=stage,
            validate=request.validate,
        )
    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    return EvaluateAnswerResponse(
        question=evaluation.question,
        answer=evaluation.answer,
        stage=evaluation.stage.value,
        scores=EvaluationScoresResponse(
            technical_accuracy=evaluation.scores.technical_accuracy,
            completeness=evaluation.scores.completeness,
            clarity=evaluation.scores.clarity,
            depth=evaluation.scores.depth,
            overall=evaluation.scores.overall,
        ),
        strengths=evaluation.strengths,
        improvements=evaluation.improvements,
        follow_up_question=evaluation.follow_up_question,
        recommendation=evaluation.recommendation.value,
        notes=evaluation.notes,
        is_validated=evaluation.is_validated,
        validation_issues=evaluation.validation_issues,
    )


@router.post("/answers/evaluate-behavioral", response_model=EvaluateAnswerResponse)
async def evaluate_behavioral_answer(request: EvaluateBehavioralRequest):
    """
    Evaluate a behavioral interview answer using STAR criteria.
    
    Assesses Situation, Task, Action, Result components and
    checks for red/green flags.
    """
    evaluator = get_answer_evaluator()
    
    try:
        evaluation = await evaluator.evaluate_behavioral_answer(
            question=request.question,
            answer=request.answer,
            competency=request.competency,
            red_flags=request.red_flags,
            green_flags=request.green_flags,
            validate=request.validate,
        )
    except Exception as e:
        logger.error(f"Behavioral evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    return EvaluateAnswerResponse(
        question=evaluation.question,
        answer=evaluation.answer,
        stage=evaluation.stage.value,
        scores=EvaluationScoresResponse(
            technical_accuracy=evaluation.scores.technical_accuracy,
            completeness=evaluation.scores.completeness,
            clarity=evaluation.scores.clarity,
            depth=evaluation.scores.depth,
            overall=evaluation.scores.overall,
        ),
        strengths=evaluation.strengths,
        improvements=evaluation.improvements,
        follow_up_question=evaluation.follow_up_question,
        recommendation=evaluation.recommendation.value,
        notes=evaluation.notes,
        is_validated=evaluation.is_validated,
        validation_issues=evaluation.validation_issues,
    )


@router.post("/questions/follow-up", response_model=FollowUpResponse)
async def generate_follow_up(request: GenerateFollowUpRequest):
    """
    Generate a follow-up question based on candidate's answer.
    
    Analyzes the answer and generates an appropriate follow-up
    to probe deeper or clarify.
    """
    generator = get_question_generator()
    
    try:
        follow_up = await generator.generate_follow_up(
            original_question=request.original_question,
            candidate_answer=request.candidate_answer,
            evaluation_summary=request.evaluation_summary,
        )
    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Follow-up generation failed: {str(e)}")
    
    return FollowUpResponse(follow_up_question=follow_up)


@router.get("/questions/stages")
async def list_interview_stages():
    """List available interview stages."""
    return {
        "stages": [
            {"value": s.value, "name": s.name}
            for s in InterviewStage
        ]
    }


@router.get("/questions/difficulties")
async def list_difficulties():
    """List available difficulty levels."""
    return {
        "difficulties": [
            {"value": d.value, "name": d.name}
            for d in QuestionDifficulty
        ]
    }
