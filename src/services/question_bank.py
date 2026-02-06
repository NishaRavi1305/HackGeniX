"""
Question Bank Service.

Provides lazy-loading, parsing, and enrichment of the curated question bank.
Supports JSONL format with automatic metadata enrichment.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.models.question_bank import (
    BankQuestion,
    QuestionCategory,
    QuestionDifficulty,
    InterviewStageHint,
    DomainInfo,
    QuestionBankStats,
    DOMAIN_KEYWORDS,
)

logger = logging.getLogger(__name__)

# Default path to question bank
DEFAULT_QUESTION_BANK_PATH = Path(__file__).parent.parent.parent / "questionBank" / "domains"


# Regex patterns for category detection
CATEGORY_PATTERNS: List[Tuple[re.Pattern, QuestionCategory]] = [
    (re.compile(r"^Explain\s+", re.IGNORECASE), QuestionCategory.EXPLAIN),
    (re.compile(r"^Design\s+(a\s+)?(solution|system)", re.IGNORECASE), QuestionCategory.DESIGN),
    (re.compile(r"^Compare\s+two\s+approaches", re.IGNORECASE), QuestionCategory.COMPARE),
    (re.compile(r"^Troubleshoot\s+", re.IGNORECASE), QuestionCategory.TROUBLESHOOT),
    (re.compile(r"performance\s+bottlenecks?", re.IGNORECASE), QuestionCategory.PERFORMANCE),
    (re.compile(r"real-?world\s+example", re.IGNORECASE), QuestionCategory.REAL_WORLD),
    (re.compile(r"scale\s+.*from\s+1[×x]", re.IGNORECASE), QuestionCategory.SCALE),
    (re.compile(r"^How\s+would\s+you\s+secure", re.IGNORECASE), QuestionCategory.SECURITY),
    (re.compile(r"^How\s+would\s+you\s+test", re.IGNORECASE), QuestionCategory.TESTING),
    (re.compile(r"(metrics|SLI).*capture\s+success", re.IGNORECASE), QuestionCategory.METRICS),
    (re.compile(r"rollout\s+strategy", re.IGNORECASE), QuestionCategory.ROLLOUT),
    (re.compile(r"trade-?offs.*PM|PM.*trade-?offs", re.IGNORECASE), QuestionCategory.TRADEOFFS),
    (re.compile(r"mentor.*junior|junior.*mentor", re.IGNORECASE), QuestionCategory.MENTORING),
    (re.compile(r"reduce\s+cost|cost.*without\s+harming", re.IGNORECASE), QuestionCategory.COST),
    (re.compile(r"telemetry.*add|add.*telemetry", re.IGNORECASE), QuestionCategory.TELEMETRY),
    (re.compile(r"^Walk\s+through\s+(a\s+)?minimal", re.IGNORECASE), QuestionCategory.WALKTHROUGH),
]

# Difficulty indicators
HARD_INDICATORS = [
    "scale from 1", "100x", "production-grade", "production-ready",
    "end-to-end architecture", "scaling considerations",
    "distributed", "high availability", "fault tolerance",
]
EASY_INDICATORS = [
    "explain", "what is", "describe", "basic", "simple",
    "give an example", "introduce",
]

# Common technical skills for extraction
SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "ruby", "php", "swift", "kotlin", "scala",
    # Frameworks
    "fastapi", "django", "flask", "spring", "express", "react", "vue",
    "angular", "nextjs", "rails", "laravel",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "sql server",
    # Infrastructure
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "terraform",
    "ansible", "jenkins", "github actions", "gitlab ci",
    # Concepts
    "microservices", "api", "rest", "graphql", "grpc", "websocket",
    "caching", "queue", "kafka", "rabbitmq", "pub/sub",
    "ci/cd", "devops", "sre", "monitoring", "observability",
    "load balancing", "sharding", "replication", "indexing",
    # ML/AI
    "machine learning", "deep learning", "neural network", "nlp",
    "transformers", "rag", "embeddings", "tensorflow", "pytorch",
    # Security
    "authentication", "authorization", "oauth", "jwt", "encryption",
    "owasp", "security", "ssl", "tls",
    # General
    "algorithms", "data structures", "oop", "design patterns",
    "testing", "debugging", "profiling", "concurrency", "threading",
]


class QuestionBankService:
    """
    Service for loading and managing the curated question bank.
    
    Features:
    - Lazy loading of domains on demand
    - JSONL format parsing
    - Automatic metadata enrichment (category, difficulty, skills)
    - Deduplication of questions
    - In-memory caching of loaded domains
    """
    
    def __init__(self, bank_path: Optional[Path] = None):
        """
        Initialize the question bank service.
        
        Args:
            bank_path: Path to the question bank directory.
                       Defaults to questionBank/domains/
        """
        self.bank_path = bank_path or DEFAULT_QUESTION_BANK_PATH
        
        # Cache of loaded domains: domain_name -> list of questions
        self._loaded_domains: Dict[str, List[BankQuestion]] = {}
        
        # Index for fast lookup
        self._by_skill: Dict[str, Set[str]] = defaultdict(set)  # skill -> question IDs
        self._by_category: Dict[QuestionCategory, Set[str]] = defaultdict(set)
        
        # Track available domains
        self._available_domains: Optional[List[str]] = None
    
    def list_available_domains(self) -> List[str]:
        """List all available domains (JSONL files in the bank directory)."""
        if self._available_domains is not None:
            return self._available_domains
        
        if not self.bank_path.exists():
            logger.warning(f"Question bank path does not exist: {self.bank_path}")
            return []
        
        domains = []
        for file_path in self.bank_path.glob("*.jsonl"):
            domain_name = file_path.stem  # filename without extension
            domains.append(domain_name)
        
        self._available_domains = sorted(domains)
        logger.info(f"Found {len(domains)} available domains: {domains}")
        return self._available_domains
    
    def get_loaded_domains(self) -> List[str]:
        """Get list of currently loaded domains."""
        return list(self._loaded_domains.keys())
    
    def is_domain_loaded(self, domain: str) -> bool:
        """Check if a domain is already loaded."""
        return domain in self._loaded_domains
    
    async def load_domain(self, domain: str) -> List[BankQuestion]:
        """
        Load a single domain from the question bank.
        
        Args:
            domain: Domain name (e.g., "backend", "aiml")
            
        Returns:
            List of BankQuestion objects
        """
        if domain in self._loaded_domains:
            logger.debug(f"Domain '{domain}' already loaded, using cache")
            return self._loaded_domains[domain]
        
        file_path = self.bank_path / f"{domain}.jsonl"
        if not file_path.exists():
            logger.warning(f"Domain file not found: {file_path}")
            return []
        
        questions = await self._parse_jsonl_file(file_path, domain)
        
        # Deduplicate
        unique_questions = list(set(questions))
        if len(unique_questions) < len(questions):
            logger.info(f"Deduplicated {len(questions) - len(unique_questions)} duplicate questions in '{domain}'")
        
        # Enrich with metadata
        enriched_questions = [self._enrich_question(q) for q in unique_questions]
        
        # Cache
        self._loaded_domains[domain] = enriched_questions
        
        # Update indexes
        self._update_indexes(enriched_questions)
        
        logger.info(f"Loaded {len(enriched_questions)} questions from domain '{domain}'")
        return enriched_questions
    
    async def load_domains(self, domains: List[str]) -> Dict[str, List[BankQuestion]]:
        """
        Load multiple domains.
        
        Args:
            domains: List of domain names
            
        Returns:
            Dict mapping domain name to list of questions
        """
        result = {}
        for domain in domains:
            result[domain] = await self.load_domain(domain)
        return result
    
    async def load_all_domains(self) -> Dict[str, List[BankQuestion]]:
        """Load all available domains."""
        available = self.list_available_domains()
        return await self.load_domains(available)
    
    async def _parse_jsonl_file(self, file_path: Path, domain: str) -> List[BankQuestion]:
        """Parse a JSONL file into BankQuestion objects."""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        question_text = data.get("question", "")
                        if not question_text:
                            continue
                        
                        questions.append(BankQuestion(
                            domain=data.get("domain", domain),
                            question_text=question_text,
                            source_file=file_path.name,
                        ))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        return questions
    
    def _enrich_question(self, question: BankQuestion) -> BankQuestion:
        """
        Enrich a question with metadata.
        
        Detects:
        - Category (from question pattern)
        - Difficulty (from complexity indicators)
        - Skills (extracted keywords)
        - Stage hint (inferred from domain and category)
        """
        text = question.question_text
        
        # Detect category
        question.category = self._detect_category(text)
        
        # Detect difficulty
        question.difficulty = self._detect_difficulty(text)
        
        # Extract skills
        question.skills = self._extract_skills(text)
        
        # Infer stage hint
        question.stage_hint = self._infer_stage_hint(question)
        
        return question
    
    def _detect_category(self, text: str) -> QuestionCategory:
        """Detect the category of a question from its text."""
        for pattern, category in CATEGORY_PATTERNS:
            if pattern.search(text):
                return category
        return QuestionCategory.GENERAL
    
    def _detect_difficulty(self, text: str) -> QuestionDifficulty:
        """Infer difficulty from question text."""
        text_lower = text.lower()
        
        # Check for hard indicators
        hard_score = sum(1 for indicator in HARD_INDICATORS if indicator in text_lower)
        
        # Check for easy indicators
        easy_score = sum(1 for indicator in EASY_INDICATORS if indicator in text_lower)
        
        # Check question length (longer questions tend to be harder)
        word_count = len(text.split())
        
        if hard_score >= 2 or (hard_score >= 1 and word_count > 25):
            return QuestionDifficulty.HARD
        elif easy_score >= 2 or (easy_score >= 1 and word_count < 15):
            return QuestionDifficulty.EASY
        else:
            return QuestionDifficulty.MEDIUM
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skill keywords from question text."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in SKILL_KEYWORDS:
            if skill in text_lower:
                found_skills.append(skill)
        
        # Also extract topic from question structure
        # e.g., "Explain microservices in the context of backend"
        topic_match = re.search(r"Explain\s+(\w+(?:\s+\w+)?)\s+in", text, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).lower()
            if topic not in found_skills:
                found_skills.append(topic)
        
        # Extract from "related to X" pattern
        related_match = re.search(r"related to\s+(\w+(?:\s+\w+)?)", text, re.IGNORECASE)
        if related_match:
            topic = related_match.group(1).lower()
            if topic not in found_skills:
                found_skills.append(topic)
        
        return found_skills[:10]  # Limit to 10 skills
    
    def _infer_stage_hint(self, question: BankQuestion) -> InterviewStageHint:
        """Infer which interview stage a question fits best."""
        domain = question.domain
        category = question.category
        
        # System design domain -> system_design stage
        if domain == "system_design":
            return InterviewStageHint.SYSTEM_DESIGN
        
        # Behavioral categories
        if category in [QuestionCategory.MENTORING, QuestionCategory.TRADEOFFS]:
            return InterviewStageHint.BEHAVIORAL
        
        # Design questions are often system design
        if category == QuestionCategory.DESIGN and question.difficulty == QuestionDifficulty.HARD:
            return InterviewStageHint.SYSTEM_DESIGN
        
        # Most questions are technical
        return InterviewStageHint.TECHNICAL
    
    def _update_indexes(self, questions: List[BankQuestion]) -> None:
        """Update lookup indexes with new questions."""
        for q in questions:
            # Skill index
            for skill in q.skills:
                self._by_skill[skill.lower()].add(q.id)
            
            # Category index
            self._by_category[q.category].add(q.id)
    
    def get_questions_by_skill(
        self,
        skill: str,
        domains: Optional[List[str]] = None,
    ) -> List[BankQuestion]:
        """
        Get questions that involve a specific skill.
        
        Args:
            skill: Skill to search for
            domains: Optional list of domains to search in
            
        Returns:
            List of matching questions
        """
        skill_lower = skill.lower()
        question_ids = self._by_skill.get(skill_lower, set())
        
        result = []
        for domain, questions in self._loaded_domains.items():
            if domains and domain not in domains:
                continue
            for q in questions:
                if q.id in question_ids or skill_lower in [s.lower() for s in q.skills]:
                    result.append(q)
        
        return result
    
    def get_questions_by_category(
        self,
        category: QuestionCategory,
        domains: Optional[List[str]] = None,
    ) -> List[BankQuestion]:
        """
        Get questions of a specific category.
        
        Args:
            category: Question category
            domains: Optional list of domains to search in
            
        Returns:
            List of matching questions
        """
        result = []
        for domain, questions in self._loaded_domains.items():
            if domains and domain not in domains:
                continue
            for q in questions:
                if q.category == category:
                    result.append(q)
        
        return result
    
    def get_all_loaded_questions(self) -> List[BankQuestion]:
        """Get all questions from loaded domains."""
        all_questions = []
        for questions in self._loaded_domains.values():
            all_questions.extend(questions)
        return all_questions
    
    def get_domain_info(self, domain: str) -> Optional[DomainInfo]:
        """Get information about a loaded domain."""
        if domain not in self._loaded_domains:
            return None
        
        questions = self._loaded_domains[domain]
        
        # Count categories
        categories: Dict[str, int] = defaultdict(int)
        for q in questions:
            categories[q.category.value] += 1
        
        # Count difficulties
        difficulties: Dict[str, int] = defaultdict(int)
        for q in questions:
            difficulties[q.difficulty.value] += 1
        
        # Collect skills
        all_skills: Set[str] = set()
        for q in questions:
            all_skills.update(q.skills)
        
        return DomainInfo(
            name=domain,
            question_count=len(questions),
            categories=dict(categories),
            difficulties=dict(difficulties),
            skills=sorted(all_skills),
            source_file=f"{domain}.jsonl",
        )
    
    def get_stats(self) -> QuestionBankStats:
        """Get statistics about the question bank."""
        all_questions = self.get_all_loaded_questions()
        
        # Questions by domain
        by_domain = {domain: len(qs) for domain, qs in self._loaded_domains.items()}
        
        # Questions by category
        by_category: Dict[str, int] = defaultdict(int)
        for q in all_questions:
            by_category[q.category.value] += 1
        
        # Questions by difficulty
        by_difficulty: Dict[str, int] = defaultdict(int)
        for q in all_questions:
            by_difficulty[q.difficulty.value] += 1
        
        # Unique skills
        all_skills: Set[str] = set()
        for q in all_questions:
            all_skills.update(q.skills)
        
        return QuestionBankStats(
            total_questions=len(all_questions),
            loaded_domains=self.get_loaded_domains(),
            available_domains=self.list_available_domains(),
            questions_by_domain=by_domain,
            questions_by_category=dict(by_category),
            questions_by_difficulty=dict(by_difficulty),
            unique_skills=sorted(all_skills),
        )
    
    def search_questions(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        category: Optional[QuestionCategory] = None,
        difficulty: Optional[QuestionDifficulty] = None,
        limit: int = 20,
    ) -> List[BankQuestion]:
        """
        Search questions by text and filters.
        
        Args:
            query: Text to search for in questions
            domains: Optional domain filter
            category: Optional category filter
            difficulty: Optional difficulty filter
            limit: Maximum results to return
            
        Returns:
            List of matching questions
        """
        query_lower = query.lower()
        results = []
        
        for domain, questions in self._loaded_domains.items():
            if domains and domain not in domains:
                continue
            
            for q in questions:
                # Apply filters
                if category and q.category != category:
                    continue
                if difficulty and q.difficulty != difficulty:
                    continue
                
                # Text search
                if query_lower in q.question_text.lower():
                    results.append(q)
                    if len(results) >= limit:
                        return results
        
        return results


# Global instance (lazy loaded)
_question_bank_service: Optional[QuestionBankService] = None


def get_question_bank_service() -> QuestionBankService:
    """Get or create the question bank service instance."""
    global _question_bank_service
    if _question_bank_service is None:
        _question_bank_service = QuestionBankService()
    return _question_bank_service
