"""
Hybrid Question Selector.

Selects questions from the curated bank and identifies gaps for LLM generation.
Ensures diversity across categories and skills while matching JD requirements.
"""
import logging
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.models.question_bank import (
    BankQuestion,
    QuestionCategory,
    QuestionDifficulty,
    InterviewStageHint,
    QuestionBankConfig,
    detect_domains_from_text,
)
from src.models.documents import ParsedResume, ParsedJobDescription
from src.services.question_bank import QuestionBankService, get_question_bank_service

logger = logging.getLogger(__name__)


class HybridQuestionSelector:
    """
    Selects questions from the bank and identifies gaps for LLM generation.
    
    The selector:
    1. Auto-detects relevant domains from JD (or uses admin-specified domains)
    2. Matches JD skills to bank questions
    3. Ensures diversity across categories
    4. Identifies skills not covered by bank for LLM gap-filling
    5. Returns a mix of bank questions and gap-filling targets
    """
    
    def __init__(self, bank_service: Optional[QuestionBankService] = None):
        """
        Initialize the selector.
        
        Args:
            bank_service: Question bank service instance
        """
        self.bank_service = bank_service or get_question_bank_service()
    
    async def select_questions(
        self,
        jd: ParsedJobDescription,
        resume: ParsedResume,
        config: QuestionBankConfig,
        stage: InterviewStageHint,
        count: int,
    ) -> Tuple[List[BankQuestion], List[str]]:
        """
        Select questions from the bank for an interview.
        
        Args:
            jd: Parsed job description
            resume: Parsed candidate resume
            config: Question bank configuration
            stage: Interview stage to select questions for
            count: Total number of questions needed
            
        Returns:
            Tuple of:
                - List of selected bank questions
                - List of skills not covered (for LLM gap-filling)
        """
        if not config.use_question_bank:
            # Bank disabled, return empty - all questions will be LLM generated
            all_skills = self._extract_jd_skills(jd)
            return [], all_skills
        
        # Determine domains to use
        domains = await self._determine_domains(jd, config)
        
        if not domains:
            logger.warning("No domains available for question selection")
            all_skills = self._extract_jd_skills(jd)
            return [], all_skills
        
        # Load required domains
        await self.bank_service.load_domains(domains)
        
        # Get all questions from loaded domains
        all_questions = []
        for domain in domains:
            questions = self.bank_service._loaded_domains.get(domain, [])
            all_questions.extend(questions)
        
        if not all_questions:
            logger.warning(f"No questions found in domains: {domains}")
            all_skills = self._extract_jd_skills(jd)
            return [], all_skills
        
        # Filter by stage
        stage_questions = self._filter_by_stage(all_questions, stage)
        
        # Filter by difficulty if specified
        if config.difficulty_filter:
            stage_questions = [q for q in stage_questions if q.difficulty == config.difficulty_filter]
        
        # Filter by category exclusions/inclusions
        stage_questions = self._filter_by_categories(stage_questions, config)
        
        # Extract skills from JD
        jd_skills = self._extract_jd_skills(jd)
        resume_skills = self._extract_resume_skills(resume)
        
        # Calculate how many questions to select from bank
        bank_count = int(count * config.bank_question_ratio)
        bank_count = min(bank_count, len(stage_questions))
        
        # Select questions with skill matching and diversity
        selected_questions = self._select_diverse_questions(
            questions=stage_questions,
            jd_skills=jd_skills,
            resume_skills=resume_skills,
            count=bank_count,
            config=config,
        )
        
        # Identify uncovered skills for gap-filling
        covered_skills = set()
        for q in selected_questions:
            covered_skills.update(s.lower() for s in q.skills)
        
        uncovered_skills = [
            skill for skill in jd_skills
            if skill.lower() not in covered_skills
        ]
        
        logger.info(
            f"Selected {len(selected_questions)} bank questions for stage '{stage.value}'. "
            f"Uncovered skills for LLM: {uncovered_skills[:5]}..."
        )
        
        return selected_questions, uncovered_skills
    
    async def _determine_domains(
        self,
        jd: ParsedJobDescription,
        config: QuestionBankConfig,
    ) -> List[str]:
        """Determine which domains to use for question selection."""
        available_domains = self.bank_service.list_available_domains()
        
        if not available_domains:
            return []
        
        # If admin specified domains, use those (filtered by availability)
        if config.enabled_domains:
            specified = [d for d in config.enabled_domains if d in available_domains]
            if specified:
                logger.info(f"Using admin-specified domains: {specified}")
                return specified
            logger.warning(
                f"Specified domains {config.enabled_domains} not found. "
                f"Available: {available_domains}"
            )
        
        # Auto-detect from JD
        if config.auto_detect_domains:
            jd_text = self._get_jd_text(jd)
            detected = detect_domains_from_text(jd_text, top_n=3)
            
            # Filter by available domains
            detected = [d for d in detected if d in available_domains]
            
            if detected:
                logger.info(f"Auto-detected domains from JD: {detected}")
                return detected
        
        # Fallback: use software_engineer and backend as defaults
        defaults = ["software_engineer", "backend"]
        fallback = [d for d in defaults if d in available_domains]
        if fallback:
            logger.info(f"Using fallback domains: {fallback}")
            return fallback
        
        # Last resort: use first available domain
        return available_domains[:1]
    
    def _get_jd_text(self, jd: ParsedJobDescription) -> str:
        """Extract searchable text from JD."""
        parts = [
            jd.title or "",
            jd.raw_text or "",
            " ".join(jd.required_skills or []),
            " ".join(jd.preferred_skills or []),
            " ".join(jd.responsibilities or []),
            " ".join(jd.qualifications or []),
        ]
        return " ".join(parts)
    
    def _extract_jd_skills(self, jd: ParsedJobDescription) -> List[str]:
        """Extract skills from job description."""
        skills = []
        if jd.required_skills:
            skills.extend(jd.required_skills)
        if jd.preferred_skills:
            skills.extend(jd.preferred_skills)
        
        # Deduplicate while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _extract_resume_skills(self, resume: ParsedResume) -> List[str]:
        """Extract skills from resume."""
        return resume.skills or []
    
    def _filter_by_stage(
        self,
        questions: List[BankQuestion],
        stage: InterviewStageHint,
    ) -> List[BankQuestion]:
        """Filter questions by interview stage."""
        # Map stage to acceptable stage hints
        stage_map = {
            InterviewStageHint.SCREENING: [
                InterviewStageHint.GENERAL,
                InterviewStageHint.SCREENING,
            ],
            InterviewStageHint.TECHNICAL: [
                InterviewStageHint.TECHNICAL,
                InterviewStageHint.GENERAL,
            ],
            InterviewStageHint.BEHAVIORAL: [
                InterviewStageHint.BEHAVIORAL,
                InterviewStageHint.GENERAL,
            ],
            InterviewStageHint.SYSTEM_DESIGN: [
                InterviewStageHint.SYSTEM_DESIGN,
                InterviewStageHint.TECHNICAL,  # Technical questions can work for system design
            ],
            InterviewStageHint.GENERAL: [
                InterviewStageHint.GENERAL,
                InterviewStageHint.TECHNICAL,
                InterviewStageHint.BEHAVIORAL,
            ],
        }
        
        acceptable_stages = stage_map.get(stage, [stage])
        return [q for q in questions if q.stage_hint in acceptable_stages]
    
    def _filter_by_categories(
        self,
        questions: List[BankQuestion],
        config: QuestionBankConfig,
    ) -> List[BankQuestion]:
        """Filter questions by category inclusions/exclusions."""
        filtered = questions
        
        # Apply exclusions
        if config.exclude_categories:
            filtered = [q for q in filtered if q.category not in config.exclude_categories]
        
        # Apply inclusions (if specified, only include these)
        if config.include_categories:
            filtered = [q for q in filtered if q.category in config.include_categories]
        
        return filtered
    
    def _select_diverse_questions(
        self,
        questions: List[BankQuestion],
        jd_skills: List[str],
        resume_skills: List[str],
        count: int,
        config: QuestionBankConfig,
    ) -> List[BankQuestion]:
        """
        Select diverse questions that match JD skills.
        
        Selection strategy:
        1. Score questions by skill relevance
        2. Ensure category variety
        3. Prefer questions matching both JD and resume skills
        """
        if not questions:
            return []
        
        if len(questions) <= count:
            return questions
        
        # Score each question
        scored_questions: List[Tuple[BankQuestion, float]] = []
        jd_skills_lower = {s.lower() for s in jd_skills}
        resume_skills_lower = {s.lower() for s in resume_skills}
        
        for q in questions:
            score = 0.0
            q_skills_lower = {s.lower() for s in q.skills}
            
            # JD skill match (higher weight)
            jd_matches = len(q_skills_lower & jd_skills_lower)
            score += jd_matches * 2.0
            
            # Resume skill match (bonus for personalization opportunity)
            resume_matches = len(q_skills_lower & resume_skills_lower)
            score += resume_matches * 1.0
            
            # Prefer medium difficulty for balance
            if q.difficulty == QuestionDifficulty.MEDIUM:
                score += 0.5
            
            # Small random factor for variety
            score += random.random() * 0.3
            
            scored_questions.append((q, score))
        
        # Sort by score descending
        scored_questions.sort(key=lambda x: x[1], reverse=True)
        
        # Select with category diversity
        selected: List[BankQuestion] = []
        category_counts: Dict[QuestionCategory, int] = defaultdict(int)
        max_per_category = config.max_questions_per_category
        
        for q, score in scored_questions:
            if len(selected) >= count:
                break
            
            # Check category limit if diversity is enabled
            if config.ensure_category_variety:
                if category_counts[q.category] >= max_per_category:
                    continue
            
            selected.append(q)
            category_counts[q.category] += 1
        
        # If we didn't get enough due to category limits, fill with remaining
        if len(selected) < count:
            remaining = [q for q, _ in scored_questions if q not in selected]
            needed = count - len(selected)
            selected.extend(remaining[:needed])
        
        return selected
    
    def get_skill_coverage_report(
        self,
        selected_questions: List[BankQuestion],
        jd_skills: List[str],
    ) -> Dict[str, any]:
        """
        Generate a report on skill coverage.
        
        Returns:
            Dict with coverage statistics
        """
        covered_skills: Set[str] = set()
        for q in selected_questions:
            covered_skills.update(s.lower() for s in q.skills)
        
        jd_skills_lower = {s.lower() for s in jd_skills}
        
        covered = covered_skills & jd_skills_lower
        uncovered = jd_skills_lower - covered_skills
        
        return {
            "total_jd_skills": len(jd_skills),
            "covered_skills": list(covered),
            "uncovered_skills": list(uncovered),
            "coverage_percent": (len(covered) / len(jd_skills) * 100) if jd_skills else 0,
            "questions_selected": len(selected_questions),
            "categories_used": list(set(q.category.value for q in selected_questions)),
        }


# Global instance
_hybrid_selector: Optional[HybridQuestionSelector] = None


def get_hybrid_question_selector() -> HybridQuestionSelector:
    """Get or create the hybrid question selector instance."""
    global _hybrid_selector
    if _hybrid_selector is None:
        _hybrid_selector = HybridQuestionSelector()
    return _hybrid_selector
