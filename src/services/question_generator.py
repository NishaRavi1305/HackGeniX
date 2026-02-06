"""
Question Generation Service.

Generates interview questions using LLM based on job description,
candidate resume, and interview stage.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.providers.llm import (
    BaseLLMProvider,
    Message,
    GenerationConfig,
    system_message,
    user_message,
    get_llm_provider_sync,
)
from src.services.prompts import (
    InterviewStage,
    QuestionDifficulty,
    INTERVIEWER_SYSTEM_PROMPT,
    QUESTION_GENERATION_PROMPTS,
    FOLLOW_UP_PROMPT,
    DIFFICULTY_ADJUSTMENT_PROMPT,
)
from src.models.documents import ParsedResume, ParsedJobDescription

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """A generated interview question with metadata."""
    question: str
    stage: InterviewStage
    difficulty: QuestionDifficulty
    category: Optional[str] = None
    purpose: str = ""
    expected_answer_points: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    duration_seconds: int = 120
    competency: Optional[str] = None  # For behavioral questions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "stage": self.stage.value,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "purpose": self.purpose,
            "expected_answer_points": self.expected_answer_points,
            "follow_up_questions": self.follow_up_questions,
            "duration_seconds": self.duration_seconds,
            "competency": self.competency,
        }


@dataclass
class QuestionGenerationRequest:
    """Request parameters for question generation."""
    stage: InterviewStage
    num_questions: int = 5
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    focus_areas: List[str] = field(default_factory=list)
    exclude_topics: List[str] = field(default_factory=list)


class QuestionGenerator:
    """
    Service for generating interview questions using LLM.
    
    Generates contextual questions based on:
    - Job description requirements
    - Candidate's resume and experience
    - Interview stage (screening, technical, behavioral)
    - Previous questions asked (to avoid repetition)
    """
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize the question generator.
        
        Args:
            llm_provider: LLM provider instance. If None, uses default from factory.
        """
        self.llm = llm_provider or get_llm_provider_sync()
        self._generation_config = GenerationConfig(
            max_tokens=2048,
            temperature=0.7,  # Some creativity in questions
            top_p=0.9,
        )
    
    def _build_resume_summary(self, resume: ParsedResume) -> str:
        """Build a concise summary of the resume for prompts."""
        parts = []
        
        if resume.summary:
            parts.append(f"Summary: {resume.summary[:500]}")
        
        if resume.skills:
            parts.append(f"Skills: {', '.join(resume.skills[:20])}")
        
        if resume.experience:
            exp_summary = []
            for exp in resume.experience[:3]:
                if exp.title and exp.company:
                    exp_summary.append(f"{exp.title} at {exp.company}")
            if exp_summary:
                parts.append(f"Experience: {'; '.join(exp_summary)}")
        
        if resume.education:
            edu_summary = []
            for edu in resume.education[:2]:
                if edu.degree and edu.institution:
                    edu_summary.append(f"{edu.degree} from {edu.institution}")
            if edu_summary:
                parts.append(f"Education: {'; '.join(edu_summary)}")
        
        return "\n".join(parts) if parts else "No resume details available"
    
    def _build_jd_summary(self, jd: ParsedJobDescription) -> str:
        """Build a concise summary of the job description for prompts."""
        parts = []
        
        if jd.title:
            parts.append(f"Role: {jd.title}")
        
        if jd.required_skills:
            parts.append(f"Required Skills: {', '.join(jd.required_skills[:15])}")
        
        if jd.preferred_skills:
            parts.append(f"Preferred Skills: {', '.join(jd.preferred_skills[:10])}")
        
        if jd.experience_years_min:
            parts.append(f"Experience Required: {jd.experience_years_min}+ years")
        
        if jd.responsibilities:
            parts.append(f"Key Responsibilities: {'; '.join(jd.responsibilities[:5])}")
        
        return "\n".join(parts) if parts else "No JD details available"
    
    def _parse_llm_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Try to find JSON array in response
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
        
        # Try to find array bounds
        if not response.startswith("["):
            start = response.find("[")
            if start != -1:
                response = response[start:]
        
        if not response.endswith("]"):
            end = response.rfind("]")
            if end != -1:
                response = response[:end + 1]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return []
    
    async def generate_questions(
        self,
        request: QuestionGenerationRequest,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        previous_questions: Optional[List[str]] = None,
    ) -> List[GeneratedQuestion]:
        """
        Generate interview questions for a specific stage.
        
        Args:
            request: Question generation parameters
            resume: Parsed candidate resume
            jd: Parsed job description
            previous_questions: Previously asked questions to avoid
            
        Returns:
            List of generated questions with metadata
        """
        stage = request.stage
        
        # Get the appropriate prompt template
        if stage not in QUESTION_GENERATION_PROMPTS:
            raise ValueError(f"No prompt template for stage: {stage}")
        
        prompt_template = QUESTION_GENERATION_PROMPTS[stage]
        
        # Build context based on stage
        context = {
            "num_questions": request.num_questions,
            "role_title": jd.title or "Software Engineer",
            "jd_summary": self._build_jd_summary(jd),
            "resume_summary": self._build_resume_summary(resume),
            "required_skills": ", ".join(jd.required_skills[:15]),
            "technical_background": self._build_resume_summary(resume),
            "focus_areas": ", ".join(request.focus_areas) if request.focus_areas else "general technical skills",
            "experience_summary": self._build_resume_summary(resume),
            "competencies": "leadership, problem-solving, teamwork, communication, adaptability",
            "technical_requirements": ", ".join(jd.required_skills[:10]),
            "system_experience": "Based on resume experience",
            "complexity_level": "appropriate for candidate experience",
        }
        
        # Format the prompt
        prompt = prompt_template.format(**context)
        
        # Add exclusion note if there are previous questions
        if previous_questions:
            prompt += f"\n\nAvoid questions similar to these already asked:\n"
            for q in previous_questions[:10]:
                prompt += f"- {q}\n"
        
        # Generate questions using LLM
        messages = [
            system_message(INTERVIEWER_SYSTEM_PROMPT),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, self._generation_config)
            questions_data = self._parse_llm_json_response(response.content)
            
            # Convert to GeneratedQuestion objects
            questions = []
            for q_data in questions_data:
                if isinstance(q_data, dict) and "question" in q_data:
                    questions.append(GeneratedQuestion(
                        question=q_data["question"],
                        stage=stage,
                        difficulty=QuestionDifficulty(q_data.get("difficulty", "medium")),
                        category=q_data.get("category"),
                        purpose=q_data.get("purpose", ""),
                        expected_answer_points=q_data.get("expected_answer_points", []),
                        follow_up_questions=q_data.get("follow_up_questions", []),
                        duration_seconds=q_data.get("duration_seconds", 120),
                        competency=q_data.get("competency"),
                    ))
            
            logger.info(f"Generated {len(questions)} {stage.value} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            raise
    
    async def generate_follow_up(
        self,
        original_question: str,
        candidate_answer: str,
        evaluation_summary: str,
    ) -> Optional[str]:
        """
        Generate a follow-up question based on candidate's answer.
        
        Args:
            original_question: The question that was asked
            candidate_answer: The candidate's response
            evaluation_summary: Brief evaluation of the answer
            
        Returns:
            Follow-up question or None if not needed
        """
        prompt = FOLLOW_UP_PROMPT.format(
            original_question=original_question,
            answer=candidate_answer,
            evaluation_summary=evaluation_summary,
        )
        
        messages = [
            system_message(INTERVIEWER_SYSTEM_PROMPT),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, GenerationConfig(
                max_tokens=512,
                temperature=0.6,
            ))
            
            # Parse response
            result = self._parse_llm_json_response(f"[{response.content}]")
            if result and isinstance(result[0], dict):
                return result[0].get("follow_up_question")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up: {e}")
            return None
    
    async def adjust_difficulty(
        self,
        performance_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine appropriate difficulty for next question based on performance.
        
        Args:
            performance_summary: Summary of candidate's performance so far
            
        Returns:
            Difficulty adjustment recommendation
        """
        prompt = DIFFICULTY_ADJUSTMENT_PROMPT.format(
            questions_answered=performance_summary.get("questions_answered", 0),
            average_score=performance_summary.get("average_score", 0),
            trend=performance_summary.get("trend", "stable"),
            strong_areas=", ".join(performance_summary.get("strong_areas", [])),
            weak_areas=", ".join(performance_summary.get("weak_areas", [])),
            current_difficulty=performance_summary.get("current_difficulty", "medium"),
        )
        
        messages = [
            system_message(INTERVIEWER_SYSTEM_PROMPT),
            user_message(prompt),
        ]
        
        try:
            response = await self.llm.generate(messages, GenerationConfig(
                max_tokens=512,
                temperature=0.3,  # More deterministic
            ))
            
            result = self._parse_llm_json_response(f"[{response.content}]")
            if result and isinstance(result[0], dict):
                return result[0]
            
            return {"next_difficulty": "medium", "difficulty_change": "maintain"}
            
        except Exception as e:
            logger.error(f"Failed to adjust difficulty: {e}")
            return {"next_difficulty": "medium", "difficulty_change": "maintain"}
    
    async def generate_screening_questions(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        num_questions: int = 3,
    ) -> List[GeneratedQuestion]:
        """Convenience method for generating screening questions."""
        request = QuestionGenerationRequest(
            stage=InterviewStage.SCREENING,
            num_questions=num_questions,
            difficulty=QuestionDifficulty.EASY,
        )
        return await self.generate_questions(request, resume, jd)
    
    async def generate_technical_questions(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        num_questions: int = 5,
        focus_areas: Optional[List[str]] = None,
    ) -> List[GeneratedQuestion]:
        """Convenience method for generating technical questions."""
        request = QuestionGenerationRequest(
            stage=InterviewStage.TECHNICAL,
            num_questions=num_questions,
            difficulty=QuestionDifficulty.MEDIUM,
            focus_areas=focus_areas or [],
        )
        return await self.generate_questions(request, resume, jd)
    
    async def generate_behavioral_questions(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        num_questions: int = 3,
    ) -> List[GeneratedQuestion]:
        """Convenience method for generating behavioral questions."""
        request = QuestionGenerationRequest(
            stage=InterviewStage.BEHAVIORAL,
            num_questions=num_questions,
            difficulty=QuestionDifficulty.MEDIUM,
        )
        return await self.generate_questions(request, resume, jd)


# Global instance (lazy loaded)
_question_generator: Optional[QuestionGenerator] = None


def get_question_generator() -> QuestionGenerator:
    """Get or create the question generator instance."""
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator()
    return _question_generator
