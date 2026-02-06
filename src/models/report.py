"""
Report data models for PDF generation.

Provides structured data models for interview assessment reports.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


class RecommendationDecision(str, Enum):
    """Hiring recommendation decisions."""
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    NO_HIRE = "no_hire"
    STRONG_NO_HIRE = "strong_no_hire"


class SeverityLevel(str, Enum):
    """Severity levels for concerns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportMetadata(BaseModel):
    """Metadata for the generated report."""
    session_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_version: str = "1.0"
    generator_version: str = "phase6-v1"


class CandidateInfo(BaseModel):
    """Candidate information section."""
    name: str
    role_title: str
    interview_date: datetime
    duration_minutes: int
    email: Optional[str] = None
    

class ScoreSection(BaseModel):
    """Individual score category with visualization data."""
    category: str
    score: float
    max_score: float = 100.0
    description: Optional[str] = None
    
    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        if self.max_score == 0:
            return 0.0
        return (self.score / self.max_score) * 100
    
    @property
    def rating(self) -> str:
        """Convert score to rating label."""
        pct = self.percentage
        if pct >= 85:
            return "Excellent"
        elif pct >= 70:
            return "Good"
        elif pct >= 55:
            return "Acceptable"
        elif pct >= 40:
            return "Below Average"
        else:
            return "Poor"


class ScoreBreakdown(BaseModel):
    """Complete score breakdown for all categories."""
    overall: ScoreSection
    technical: ScoreSection
    behavioral: ScoreSection
    communication: ScoreSection
    problem_solving: Optional[ScoreSection] = None
    
    def get_all_scores(self) -> List[ScoreSection]:
        """Get all score sections as a list."""
        scores = [self.overall, self.technical, self.behavioral, self.communication]
        if self.problem_solving:
            scores.append(self.problem_solving)
        return scores


class QuestionSummary(BaseModel):
    """Summary of a single question evaluation."""
    question_id: str
    question_text: str
    answer_text: str
    stage: str  # screening, technical, behavioral, system_design
    score: float
    max_score: float = 100.0
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    key_feedback: Optional[str] = None
    recommendation: str = "acceptable"  # strong, acceptable, weak, insufficient
    
    @property
    def executive_summary(self) -> str:
        """Generate a 2-3 sentence executive summary of this evaluation."""
        rating = "excellent" if self.score >= 80 else "good" if self.score >= 60 else "needs improvement"
        
        summary_parts = [f"Score: {self.score:.0f}/100 ({rating})."]
        
        if self.strengths:
            summary_parts.append(f"Strengths: {self.strengths[0]}.")
        
        if self.improvements:
            summary_parts.append(f"Improvement: {self.improvements[0]}.")
        
        return " ".join(summary_parts)


class ReportStrength(BaseModel):
    """Documented strength with evidence."""
    title: str
    evidence: str
    impact_level: str = "medium"  # low, medium, high
    related_questions: List[str] = Field(default_factory=list)


class ReportConcern(BaseModel):
    """Documented concern or area for improvement."""
    title: str
    severity: SeverityLevel = SeverityLevel.MEDIUM
    evidence: str
    suggestion: Optional[str] = None
    related_questions: List[str] = Field(default_factory=list)


class HiringRecommendation(BaseModel):
    """Final hiring recommendation section."""
    decision: RecommendationDecision
    confidence_percent: float  # 0-100
    reasoning: str
    risk_factors: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    
    @property
    def decision_display(self) -> str:
        """Human-readable decision."""
        mapping = {
            RecommendationDecision.STRONG_HIRE: "Strong Hire",
            RecommendationDecision.HIRE: "Hire",
            RecommendationDecision.NO_HIRE: "Do Not Hire",
            RecommendationDecision.STRONG_NO_HIRE: "Strong No Hire",
        }
        return mapping.get(self.decision, str(self.decision))
    
    @property
    def decision_color(self) -> str:
        """Color code for decision (for PDF styling)."""
        mapping = {
            RecommendationDecision.STRONG_HIRE: "#22c55e",  # green
            RecommendationDecision.HIRE: "#84cc16",  # lime
            RecommendationDecision.NO_HIRE: "#f97316",  # orange
            RecommendationDecision.STRONG_NO_HIRE: "#ef4444",  # red
        }
        return mapping.get(self.decision, "#6b7280")  # gray default


class FullInterviewReport(BaseModel):
    """
    Complete interview report data for PDF generation.
    
    Aggregates all sections into a single model that the PDF
    generator can consume.
    """
    metadata: ReportMetadata
    candidate: CandidateInfo
    executive_summary: str
    scores: ScoreBreakdown
    strengths: List[ReportStrength] = Field(default_factory=list)
    concerns: List[ReportConcern] = Field(default_factory=list)
    question_evaluations: List[QuestionSummary] = Field(default_factory=list)
    recommendation: HiringRecommendation
    
    # Optional additional data
    interview_notes: Optional[str] = None
    stage_summaries: Dict[str, str] = Field(default_factory=dict)
    
    @classmethod
    def from_orchestrator_report(
        cls,
        session_id: str,
        report_response: "InterviewReportResponse",  # Forward reference
        evaluations: List[Dict[str, Any]],
    ) -> "FullInterviewReport":
        """
        Create FullInterviewReport from orchestrator's InterviewReportResponse.
        
        This is the bridge between the orchestrator output and PDF generation.
        """
        # Build metadata
        metadata = ReportMetadata(session_id=session_id)
        
        # Build candidate info
        candidate = CandidateInfo(
            name=report_response.candidate_name,
            role_title=report_response.role_title,
            interview_date=datetime.utcnow(),  # Would come from session
            duration_minutes=report_response.duration_minutes,
        )
        
        # Build score breakdown
        scores = ScoreBreakdown(
            overall=ScoreSection(
                category="Overall",
                score=report_response.overall_score,
                description="Composite score across all evaluation areas",
            ),
            technical=ScoreSection(
                category="Technical",
                score=report_response.technical_score,
                description="Technical knowledge and problem-solving ability",
            ),
            behavioral=ScoreSection(
                category="Behavioral",
                score=report_response.behavioral_score,
                description="Soft skills, teamwork, and cultural fit",
            ),
            communication=ScoreSection(
                category="Communication",
                score=report_response.communication_score,
                description="Clarity, articulation, and professional communication",
            ),
        )
        
        # Build strengths from report
        strengths = []
        for s in report_response.strengths:
            if isinstance(s, dict):
                strengths.append(ReportStrength(
                    title=s.get("title", s.get("area", "Strength")),
                    evidence=s.get("evidence", s.get("description", str(s))),
                    impact_level=s.get("impact", "medium"),
                ))
            else:
                strengths.append(ReportStrength(
                    title="Strength",
                    evidence=str(s),
                ))
        
        # Build concerns from report
        concerns = []
        for c in report_response.concerns:
            if isinstance(c, dict):
                concerns.append(ReportConcern(
                    title=c.get("title", c.get("area", "Concern")),
                    evidence=c.get("evidence", c.get("description", str(c))),
                    severity=SeverityLevel(c.get("severity", "medium")),
                    suggestion=c.get("suggestion"),
                ))
            else:
                concerns.append(ReportConcern(
                    title="Area for Improvement",
                    evidence=str(c),
                ))
        
        # Build question evaluations
        question_evals = []
        for i, e in enumerate(evaluations):
            question_evals.append(QuestionSummary(
                question_id=e.get("question_id", f"q_{i}"),
                question_text=e.get("question", ""),
                answer_text=e.get("answer", ""),
                stage=e.get("stage", "unknown"),
                score=e.get("scores", {}).get("overall", 0),
                strengths=e.get("strengths", []),
                improvements=e.get("improvements", []),
                key_feedback=e.get("notes", None),
                recommendation=e.get("recommendation", "acceptable"),
            ))
        
        # Build recommendation
        recommendation = HiringRecommendation(
            decision=RecommendationDecision(report_response.recommendation),
            confidence_percent=report_response.confidence,
            reasoning=report_response.reasoning,
            risk_factors=[],  # Could be extracted from concerns
            next_steps=report_response.next_steps,
        )
        
        return cls(
            metadata=metadata,
            candidate=candidate,
            executive_summary=report_response.executive_summary,
            scores=scores,
            strengths=strengths,
            concerns=concerns,
            question_evaluations=question_evals,
            recommendation=recommendation,
        )
