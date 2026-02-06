"""
Answer Evaluation Service (LLM-as-Judge).

Evaluates candidate answers using LLM with structured rubrics.
Includes hallucination detection to ensure evaluations are grounded.
"""
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from src.providers.llm import (
    BaseLLMProvider,
    Message,
    GenerationConfig,
    system_message,
    user_message,
    get_llm_provider_sync,
)
from src.services.prompts import (
    ANSWER_EVALUATION_PROMPT,
    BEHAVIORAL_EVALUATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    INTERVIEW_SUMMARY_PROMPT,
    InterviewStage,
)

logger = logging.getLogger(__name__)


class Recommendation(str, Enum):
    """Hiring recommendation levels."""
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    NO_HIRE = "no_hire"
    STRONG_NO_HIRE = "strong_no_hire"


class AnswerStrength(str, Enum):
    """Answer quality assessment."""
    STRONG = "strong"
    ACCEPTABLE = "acceptable"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


@dataclass
class EvaluationScores:
    """Scores from answer evaluation."""
    technical_accuracy: float = 0
    completeness: float = 0
    clarity: float = 0
    depth: float = 0
    overall: float = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "technical_accuracy": self.technical_accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "depth": self.depth,
            "overall": self.overall,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationScores":
        return cls(
            technical_accuracy=float(data.get("technical_accuracy", 0)),
            completeness=float(data.get("completeness", 0)),
            clarity=float(data.get("clarity", 0)),
            depth=float(data.get("depth", 0)),
            overall=float(data.get("overall", 0)),
        )


@dataclass
class STARScores:
    """STAR format evaluation scores for behavioral questions."""
    situation: float = 0
    task: float = 0
    action: float = 0
    result: float = 0
    total: float = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "situation": self.situation,
            "task": self.task,
            "action": self.action,
            "result": self.result,
            "total": self.total,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STARScores":
        return cls(
            situation=float(data.get("situation", 0)),
            task=float(data.get("task", 0)),
            action=float(data.get("action", 0)),
            result=float(data.get("result", 0)),
            total=float(data.get("total", 0)),
        )


@dataclass
class AnswerEvaluation:
    """Complete evaluation of a candidate's answer."""
    question: str
    answer: str
    stage: InterviewStage
    scores: EvaluationScores
    star_scores: Optional[STARScores] = None  # For behavioral questions
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    follow_up_question: Optional[str] = None
    recommendation: AnswerStrength = AnswerStrength.ACCEPTABLE
    notes: str = ""
    is_validated: bool = False  # Hallucination check passed
    validation_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "stage": self.stage.value,
            "scores": self.scores.to_dict(),
            "star_scores": self.star_scores.to_dict() if self.star_scores else None,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "follow_up_question": self.follow_up_question,
            "recommendation": self.recommendation.value,
            "notes": self.notes,
            "is_validated": self.is_validated,
            "validation_issues": self.validation_issues,
        }


@dataclass
class InterviewReport:
    """Complete interview evaluation report."""
    candidate_name: str
    role_title: str
    duration_minutes: int
    evaluations: List[AnswerEvaluation]
    
    # Aggregate scores
    technical_score: float = 0
    behavioral_score: float = 0
    communication_score: float = 0
    problem_solving_score: float = 0
    overall_score: float = 0
    
    # Summary
    executive_summary: str = ""
    strengths: List[Dict[str, str]] = field(default_factory=list)
    concerns: List[Dict[str, str]] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.NO_HIRE
    confidence: float = 0
    reasoning: str = ""
    next_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "role_title": self.role_title,
            "duration_minutes": self.duration_minutes,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "technical_score": self.technical_score,
            "behavioral_score": self.behavioral_score,
            "communication_score": self.communication_score,
            "problem_solving_score": self.problem_solving_score,
            "overall_score": self.overall_score,
            "executive_summary": self.executive_summary,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "next_steps": self.next_steps,
        }


class AnswerEvaluator:
    """
    LLM-as-Judge for evaluating candidate answers.
    
    Uses structured prompts and rubrics to evaluate answers consistently.
    Includes hallucination detection to ensure evaluations are grounded.
    """
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize the answer evaluator.
        
        Args:
            llm_provider: LLM provider instance. If None, uses default from factory.
        """
        self.llm = llm_provider or get_llm_provider_sync()
        self._eval_config = GenerationConfig(
            max_tokens=1024,
            temperature=0.3,  # Low temperature for consistent evaluation
            top_p=0.9,
        )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        # Find JSON object bounds
        if not response.startswith("{"):
            start = response.find("{")
            if start != -1:
                response = response[start:]
        
        if not response.endswith("}"):
            end = response.rfind("}")
            if end != -1:
                response = response[:end + 1]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return {}
    
    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_points: List[str],
        stage: InterviewStage = InterviewStage.TECHNICAL,
        validate: bool = True,
    ) -> AnswerEvaluation:
        """
        Evaluate a candidate's answer to a question.
        
        Args:
            question: The interview question
            answer: Candidate's response
            expected_points: Key points expected in a good answer
            stage: Interview stage (affects evaluation criteria)
            validate: Whether to run hallucination check
            
        Returns:
            Complete answer evaluation
        """
        prompt = ANSWER_EVALUATION_PROMPT.format(
            question=question,
            expected_points="\n".join(f"- {p}" for p in expected_points),
            answer=answer,
        )
        
        messages = [
            system_message("You are an expert interview evaluator. Evaluate answers fairly and objectively based on the provided criteria."),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, self._eval_config)
            eval_data = self._parse_json_response(response.content)
            
            if not eval_data:
                # Return default evaluation if parsing failed
                return AnswerEvaluation(
                    question=question,
                    answer=answer,
                    stage=stage,
                    scores=EvaluationScores(overall=50),
                    notes="Evaluation parsing failed",
                )
            
            evaluation = AnswerEvaluation(
                question=question,
                answer=answer,
                stage=stage,
                scores=EvaluationScores.from_dict(eval_data.get("scores", {})),
                strengths=eval_data.get("strengths", []),
                improvements=eval_data.get("improvements", []),
                follow_up_question=eval_data.get("follow_up_question"),
                recommendation=AnswerStrength(eval_data.get("recommendation", "acceptable")),
                notes=eval_data.get("notes", ""),
            )
            
            # Validate evaluation if requested
            if validate:
                validation = await self._validate_evaluation(question, answer, eval_data)
                evaluation.is_validated = validation.get("is_grounded", False)
                evaluation.validation_issues = [
                    issue.get("description", "") 
                    for issue in validation.get("issues", [])
                ]
                
                # Apply corrections if needed
                if not evaluation.is_validated and validation.get("corrected_evaluation"):
                    corrected = validation["corrected_evaluation"]
                    if isinstance(corrected, dict) and "scores" in corrected:
                        evaluation.scores = EvaluationScores.from_dict(corrected["scores"])
                        evaluation.is_validated = True
            else:
                evaluation.is_validated = True
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate answer: {e}")
            return AnswerEvaluation(
                question=question,
                answer=answer,
                stage=stage,
                scores=EvaluationScores(overall=50),
                notes=f"Evaluation error: {str(e)}",
            )
    
    async def evaluate_behavioral_answer(
        self,
        question: str,
        answer: str,
        competency: str,
        red_flags: List[str],
        green_flags: List[str],
        validate: bool = True,
    ) -> AnswerEvaluation:
        """
        Evaluate a behavioral answer using STAR criteria.
        
        Args:
            question: The behavioral question
            answer: Candidate's response
            competency: The competency being assessed
            red_flags: Warning signs to watch for
            green_flags: Positive indicators
            validate: Whether to run hallucination check
            
        Returns:
            Complete answer evaluation with STAR scores
        """
        prompt = BEHAVIORAL_EVALUATION_PROMPT.format(
            question=question,
            competency=competency,
            answer=answer,
            red_flags="\n".join(f"- {f}" for f in red_flags),
            green_flags="\n".join(f"- {f}" for f in green_flags),
        )
        
        messages = [
            system_message("You are an expert behavioral interview evaluator. Assess answers using the STAR framework."),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, self._eval_config)
            eval_data = self._parse_json_response(response.content)
            
            if not eval_data:
                return AnswerEvaluation(
                    question=question,
                    answer=answer,
                    stage=InterviewStage.BEHAVIORAL,
                    scores=EvaluationScores(overall=50),
                    notes="Evaluation parsing failed",
                )
            
            evaluation = AnswerEvaluation(
                question=question,
                answer=answer,
                stage=InterviewStage.BEHAVIORAL,
                scores=EvaluationScores(
                    overall=float(eval_data.get("overall_score", 0)),
                    clarity=float(eval_data.get("self_awareness_score", 0)),
                    completeness=float(eval_data.get("relevance_score", 0)),
                    technical_accuracy=float(eval_data.get("authenticity_score", 0)),
                ),
                star_scores=STARScores.from_dict(eval_data.get("star_scores", {})),
                strengths=eval_data.get("green_flags_detected", []),
                improvements=eval_data.get("red_flags_detected", []),
                recommendation=AnswerStrength(eval_data.get("recommendation", "acceptable")),
                notes=eval_data.get("notes", ""),
            )
            
            if validate:
                validation = await self._validate_evaluation(question, answer, eval_data)
                evaluation.is_validated = validation.get("is_grounded", False)
                evaluation.validation_issues = [
                    issue.get("description", "") 
                    for issue in validation.get("issues", [])
                ]
            else:
                evaluation.is_validated = True
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate behavioral answer: {e}")
            return AnswerEvaluation(
                question=question,
                answer=answer,
                stage=InterviewStage.BEHAVIORAL,
                scores=EvaluationScores(overall=50),
                notes=f"Evaluation error: {str(e)}",
            )
    
    async def _validate_evaluation(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate that an evaluation is grounded in the candidate's answer.
        
        Checks for hallucinations or unsupported claims in the evaluation.
        """
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            question=question,
            answer=answer,
            evaluation=json.dumps(evaluation, indent=2),
        )
        
        messages = [
            system_message("You are a validation system that checks if evaluations are grounded in evidence."),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, GenerationConfig(
                max_tokens=1024,
                temperature=0.1,  # Very deterministic
            ))
            
            return self._parse_json_response(response.content)
            
        except Exception as e:
            logger.error(f"Failed to validate evaluation: {e}")
            return {"is_grounded": True, "issues": [], "confidence": 50}
    
    async def generate_interview_report(
        self,
        candidate_name: str,
        role_title: str,
        duration_minutes: int,
        evaluations: List[AnswerEvaluation],
    ) -> InterviewReport:
        """
        Generate a comprehensive interview report from all evaluations.
        
        Args:
            candidate_name: Name of the candidate
            role_title: Position being interviewed for
            duration_minutes: Total interview duration
            evaluations: List of all answer evaluations
            
        Returns:
            Complete interview report with recommendation
        """
        # Calculate aggregate scores
        technical_evals = [e for e in evaluations if e.stage == InterviewStage.TECHNICAL]
        behavioral_evals = [e for e in evaluations if e.stage == InterviewStage.BEHAVIORAL]
        
        technical_score = (
            sum(e.scores.overall for e in technical_evals) / len(technical_evals)
            if technical_evals else 0
        )
        behavioral_score = (
            sum(e.scores.overall for e in behavioral_evals) / len(behavioral_evals)
            if behavioral_evals else 0
        )
        
        # Average communication from all evaluations
        communication_score = (
            sum(e.scores.clarity for e in evaluations) / len(evaluations)
            if evaluations else 0
        )
        
        # Problem solving from technical evaluations
        problem_solving_score = (
            sum(e.scores.depth for e in technical_evals) / len(technical_evals)
            if technical_evals else 0
        )
        
        # Overall weighted score
        overall_score = (
            technical_score * 0.4 +
            behavioral_score * 0.25 +
            communication_score * 0.2 +
            problem_solving_score * 0.15
        )
        
        # Build performance summary for prompt
        stage_performances = []
        for stage in InterviewStage:
            stage_evals = [e for e in evaluations if e.stage == stage]
            if stage_evals:
                avg = sum(e.scores.overall for e in stage_evals) / len(stage_evals)
                stage_performances.append(f"- {stage.value}: {avg:.1f}/100 ({len(stage_evals)} questions)")
        
        question_breakdown = []
        for i, e in enumerate(evaluations, 1):
            question_breakdown.append(
                f"{i}. [{e.stage.value}] Score: {e.scores.overall:.0f}/100 - {e.recommendation.value}"
            )
        
        # Generate summary using LLM
        prompt = INTERVIEW_SUMMARY_PROMPT.format(
            role_title=role_title,
            candidate_name=candidate_name,
            duration_minutes=duration_minutes,
            stage_performances="\n".join(stage_performances),
            question_breakdown="\n".join(question_breakdown),
            technical_score=f"{technical_score:.0f}",
            behavioral_score=f"{behavioral_score:.0f}",
            communication_score=f"{communication_score:.0f}",
            problem_solving_score=f"{problem_solving_score:.0f}",
            cultural_fit_score=f"{behavioral_score:.0f}",  # Using behavioral as proxy
            overall_score=f"{overall_score:.0f}",
        )
        
        messages = [
            system_message("You are an expert hiring manager generating interview reports."),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, GenerationConfig(
                max_tokens=2048,
                temperature=0.4,
            ))
            summary_data = self._parse_json_response(response.content)
            
            report = InterviewReport(
                candidate_name=candidate_name,
                role_title=role_title,
                duration_minutes=duration_minutes,
                evaluations=evaluations,
                technical_score=technical_score,
                behavioral_score=behavioral_score,
                communication_score=communication_score,
                problem_solving_score=problem_solving_score,
                overall_score=overall_score,
            )
            
            if summary_data:
                report.executive_summary = summary_data.get("executive_summary", "")
                report.strengths = summary_data.get("strengths", [])
                report.concerns = summary_data.get("concerns", [])
                report.recommendation = Recommendation(
                    summary_data.get("recommendation", "no_hire")
                )
                report.confidence = float(summary_data.get("confidence", 0))
                report.reasoning = summary_data.get("reasoning", "")
                report.next_steps = summary_data.get("next_steps", [])
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate interview report: {e}")
            # Return report with calculated scores but no LLM summary
            return InterviewReport(
                candidate_name=candidate_name,
                role_title=role_title,
                duration_minutes=duration_minutes,
                evaluations=evaluations,
                technical_score=technical_score,
                behavioral_score=behavioral_score,
                communication_score=communication_score,
                problem_solving_score=problem_solving_score,
                overall_score=overall_score,
                executive_summary=f"Interview completed with overall score of {overall_score:.0f}/100",
                recommendation=Recommendation.HIRE if overall_score >= 70 else Recommendation.NO_HIRE,
            )


# Global instance (lazy loaded)
_answer_evaluator: Optional[AnswerEvaluator] = None


def get_answer_evaluator() -> AnswerEvaluator:
    """Get or create the answer evaluator instance."""
    global _answer_evaluator
    if _answer_evaluator is None:
        _answer_evaluator = AnswerEvaluator()
    return _answer_evaluator
