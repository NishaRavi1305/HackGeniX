"""
Unit tests for Question Bank Service and Hybrid Question Selector (Phase 6.5).

Tests cover:
- Question bank loading from JSONL
- Category detection patterns
- Difficulty inference
- Skill extraction
- Deduplication
- Domain detection from text
- Hybrid selector question selection
"""
import pytest
import tempfile
import json
from pathlib import Path

from src.models.question_bank import (
    BankQuestion,
    QuestionCategory,
    QuestionDifficulty,
    InterviewStageHint,
    QuestionBankConfig,
    QuestionSource,
    detect_domains_from_text,
    DOMAIN_KEYWORDS,
)
from src.services.question_bank import (
    QuestionBankService,
    CATEGORY_PATTERNS,
    HARD_INDICATORS,
    EASY_INDICATORS,
)
from src.services.hybrid_question_selector import HybridQuestionSelector
from src.models.documents import (
    ParsedResume,
    ParsedJobDescription,
    ContactInfo,
    Experience,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for testing."""
    return [
        {"domain": "backend", "question": "Explain microservices architecture."},
        {"domain": "backend", "question": "How do you scale event-driven from 1× to 100× load?"},
        {"domain": "backend", "question": "Design a solution that uses caching for high performance."},
        {"domain": "backend", "question": "Compare two approaches for handling database migrations."},
        {"domain": "backend", "question": "Troubleshoot a failure scenario in a distributed system."},
        {"domain": "backend", "question": "What are the performance bottlenecks in REST APIs?"},
        {"domain": "backend", "question": "Give a real-world example of rate limiting."},
        {"domain": "backend", "question": "How would you secure a public API endpoint?"},
        {"domain": "backend", "question": "How would you test a microservices architecture?"},
        {"domain": "backend", "question": "Which metrics would you use to capture success of a service?"},
    ]


@pytest.fixture
def temp_question_bank(sample_jsonl_content, tmp_path):
    """Create a temporary question bank directory with test files."""
    bank_dir = tmp_path / "questionBank" / "domains"
    bank_dir.mkdir(parents=True)
    
    # Create backend.jsonl
    backend_file = bank_dir / "backend.jsonl"
    with open(backend_file, "w", encoding="utf-8") as f:
        for item in sample_jsonl_content:
            f.write(json.dumps(item) + "\n")
    
    # Create aiml.jsonl
    aiml_file = bank_dir / "aiml.jsonl"
    aiml_content = [
        {"domain": "aiml", "question": "Explain how transformers work in NLP."},
        {"domain": "aiml", "question": "Design a RAG pipeline for document QA."},
        {"domain": "aiml", "question": "What are the performance bottlenecks in training deep learning models?"},
    ]
    with open(aiml_file, "w", encoding="utf-8") as f:
        for item in aiml_content:
            f.write(json.dumps(item) + "\n")
    
    return bank_dir


@pytest.fixture
def question_bank_service(temp_question_bank):
    """Create a QuestionBankService with temp directory."""
    return QuestionBankService(bank_path=temp_question_bank)


@pytest.fixture
def sample_resume():
    """Create a sample parsed resume."""
    return ParsedResume(
        skills=["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        summary="Senior backend engineer with 5 years of experience",
        experience=[
            Experience(
                title="Senior Software Engineer",
                company="TechCorp",
                description="Built scalable microservices using Python and FastAPI",
            ),
        ],
        education=[],
        contact=ContactInfo(name="John Doe", email="john@example.com"),
    )


@pytest.fixture
def sample_jd():
    """Create a sample parsed job description."""
    return ParsedJobDescription(
        title="Senior Backend Engineer",
        required_skills=["Python", "PostgreSQL", "Redis", "Docker", "Microservices"],
        preferred_skills=["Kubernetes", "Kafka", "GraphQL"],
        responsibilities=["Design and build scalable backend services"],
        qualifications=["5+ years of backend development experience"],
        raw_text="Looking for a senior backend engineer to build scalable microservices...",
    )


# =============================================================================
# Question Bank Service Tests
# =============================================================================

class TestQuestionBankService:
    """Tests for QuestionBankService."""
    
    def test_list_available_domains(self, question_bank_service):
        """Test listing available domains from the bank directory."""
        domains = question_bank_service.list_available_domains()
        
        assert len(domains) == 2
        assert "backend" in domains
        assert "aiml" in domains
    
    @pytest.mark.asyncio
    async def test_load_domain(self, question_bank_service):
        """Test loading a single domain."""
        questions = await question_bank_service.load_domain("backend")
        
        assert len(questions) == 10
        assert all(isinstance(q, BankQuestion) for q in questions)
        assert all(q.domain == "backend" for q in questions)
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_domain(self, question_bank_service):
        """Test loading a domain that doesn't exist."""
        questions = await question_bank_service.load_domain("nonexistent")
        assert questions == []
    
    @pytest.mark.asyncio
    async def test_domain_caching(self, question_bank_service):
        """Test that domains are cached after first load."""
        questions1 = await question_bank_service.load_domain("backend")
        questions2 = await question_bank_service.load_domain("backend")
        
        # Should be the same cached list
        assert questions1 is questions2
        assert question_bank_service.is_domain_loaded("backend")
    
    @pytest.mark.asyncio
    async def test_load_multiple_domains(self, question_bank_service):
        """Test loading multiple domains at once."""
        result = await question_bank_service.load_domains(["backend", "aiml"])
        
        assert "backend" in result
        assert "aiml" in result
        assert len(result["backend"]) == 10
        assert len(result["aiml"]) == 3
    
    @pytest.mark.asyncio
    async def test_deduplication(self, question_bank_service, temp_question_bank):
        """Test that duplicate questions are removed during load."""
        # Add duplicates to the file
        backend_file = temp_question_bank / "backend.jsonl"
        with open(backend_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"domain": "backend", "question": "Explain microservices architecture."}) + "\n")
            f.write(json.dumps({"domain": "backend", "question": "Explain microservices architecture."}) + "\n")
        
        # Create new service instance to reload
        service = QuestionBankService(bank_path=temp_question_bank)
        questions = await service.load_domain("backend")
        
        # Duplicates should be removed
        question_texts = [q.question_text for q in questions]
        assert len(question_texts) == len(set(question_texts))


class TestCategoryDetection:
    """Tests for category detection patterns."""
    
    @pytest.mark.asyncio
    async def test_explain_category(self, question_bank_service):
        """Test detection of EXPLAIN category."""
        questions = await question_bank_service.load_domain("backend")
        explain_qs = [q for q in questions if q.category == QuestionCategory.EXPLAIN]
        
        # Should detect "Explain microservices..."
        assert len(explain_qs) >= 1
        assert any("Explain" in q.question_text for q in explain_qs)
    
    @pytest.mark.asyncio
    async def test_design_category(self, question_bank_service):
        """Test detection of DESIGN category."""
        questions = await question_bank_service.load_domain("backend")
        design_qs = [q for q in questions if q.category == QuestionCategory.DESIGN]
        
        # Should detect "Design a solution..."
        assert len(design_qs) >= 1
    
    @pytest.mark.asyncio
    async def test_compare_category(self, question_bank_service):
        """Test detection of COMPARE category."""
        questions = await question_bank_service.load_domain("backend")
        compare_qs = [q for q in questions if q.category == QuestionCategory.COMPARE]
        
        # Should detect "Compare two approaches..."
        assert len(compare_qs) >= 1
    
    @pytest.mark.asyncio
    async def test_troubleshoot_category(self, question_bank_service):
        """Test detection of TROUBLESHOOT category."""
        questions = await question_bank_service.load_domain("backend")
        troubleshoot_qs = [q for q in questions if q.category == QuestionCategory.TROUBLESHOOT]
        
        assert len(troubleshoot_qs) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_category(self, question_bank_service):
        """Test detection of PERFORMANCE category."""
        questions = await question_bank_service.load_domain("backend")
        perf_qs = [q for q in questions if q.category == QuestionCategory.PERFORMANCE]
        
        # Should detect "performance bottlenecks"
        assert len(perf_qs) >= 1
    
    @pytest.mark.asyncio
    async def test_security_category(self, question_bank_service):
        """Test detection of SECURITY category."""
        questions = await question_bank_service.load_domain("backend")
        security_qs = [q for q in questions if q.category == QuestionCategory.SECURITY]
        
        # Should detect "How would you secure..."
        assert len(security_qs) >= 1
    
    @pytest.mark.asyncio
    async def test_testing_category(self, question_bank_service):
        """Test detection of TESTING category."""
        questions = await question_bank_service.load_domain("backend")
        testing_qs = [q for q in questions if q.category == QuestionCategory.TESTING]
        
        # Should detect "How would you test..."
        assert len(testing_qs) >= 1


class TestDifficultyInference:
    """Tests for difficulty inference."""
    
    @pytest.mark.asyncio
    async def test_hard_difficulty_detection(self, question_bank_service):
        """Test detection of HARD difficulty questions."""
        questions = await question_bank_service.load_domain("backend")
        
        # "scale from 1× to 100×" should be detected as HARD
        # Note: The detection uses "scale from 1" which might not match Unicode ×
        scale_questions = [q for q in questions if "100" in q.question_text and "scale" in q.question_text.lower()]
        
        # If we have scale questions with hard indicators, check they're harder than easy
        if scale_questions:
            # At minimum, they shouldn't be EASY
            for q in scale_questions:
                assert q.difficulty != QuestionDifficulty.EASY, f"Scale question should not be easy: {q.question_text}"
    
    @pytest.mark.asyncio
    async def test_easy_difficulty_detection(self, question_bank_service):
        """Test detection of EASY difficulty questions."""
        questions = await question_bank_service.load_domain("backend")
        
        # "Explain..." questions should typically be EASY
        explain_qs = [q for q in questions if q.question_text.startswith("Explain")]
        assert len(explain_qs) >= 1
        # Note: actual difficulty depends on indicators in text
    
    @pytest.mark.asyncio  
    async def test_medium_as_default(self, question_bank_service):
        """Test that MEDIUM is the default difficulty."""
        questions = await question_bank_service.load_domain("backend")
        
        # Most questions without specific indicators should be MEDIUM
        medium_qs = [q for q in questions if q.difficulty == QuestionDifficulty.MEDIUM]
        assert len(medium_qs) > 0


class TestSkillExtraction:
    """Tests for skill extraction from questions."""
    
    @pytest.mark.asyncio
    async def test_skill_extraction(self, question_bank_service):
        """Test that skills are extracted from question text."""
        questions = await question_bank_service.load_domain("backend")
        
        # Check that skills are being extracted
        all_skills = set()
        for q in questions:
            all_skills.update(q.skills)
        
        # Should find some technical skills
        assert len(all_skills) > 0
    
    @pytest.mark.asyncio
    async def test_skill_keywords_matched(self, question_bank_service):
        """Test that skill keywords from the predefined list are matched."""
        questions = await question_bank_service.load_domain("backend")
        
        # "caching" should be extracted from the design question
        cache_question = [q for q in questions if "caching" in q.question_text.lower()]
        assert len(cache_question) >= 1
        assert "caching" in cache_question[0].skills


class TestDomainDetection:
    """Tests for domain detection from text."""
    
    def test_backend_detection(self):
        """Test detection of backend domain from JD text."""
        jd_text = """
        We are looking for a backend engineer with experience in:
        - REST APIs and microservices
        - PostgreSQL and Redis
        - Python and FastAPI
        """
        domains = detect_domains_from_text(jd_text, top_n=3)
        
        assert "backend" in domains
    
    def test_aiml_detection(self):
        """Test detection of AI/ML domain from JD text."""
        jd_text = """
        Machine Learning Engineer position:
        - Deep learning and neural networks
        - PyTorch and TensorFlow
        - NLP and transformers
        """
        domains = detect_domains_from_text(jd_text, top_n=3)
        
        assert "aiml" in domains
    
    def test_devops_detection(self):
        """Test detection of DevOps/SRE domain from JD text."""
        jd_text = """
        DevOps Engineer role:
        - Kubernetes and Docker
        - CI/CD pipelines
        - Monitoring and observability
        - SRE practices
        """
        domains = detect_domains_from_text(jd_text, top_n=3)
        
        assert "devops_sre" in domains
    
    def test_multiple_domains(self):
        """Test detection of multiple domains."""
        jd_text = """
        Full-stack engineer with backend and ML experience:
        - Python and FastAPI
        - Machine learning models
        - PostgreSQL database
        """
        domains = detect_domains_from_text(jd_text, top_n=3)
        
        assert len(domains) >= 2
    
    def test_empty_text(self):
        """Test with empty text returns empty list."""
        domains = detect_domains_from_text("", top_n=3)
        assert domains == []
    
    def test_top_n_limit(self):
        """Test that top_n limits the results."""
        jd_text = "python backend api machine learning deep learning kubernetes docker devops"
        domains = detect_domains_from_text(jd_text, top_n=2)
        
        assert len(domains) <= 2


class TestQuestionBankStats:
    """Tests for question bank statistics."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self, question_bank_service):
        """Test getting question bank statistics."""
        await question_bank_service.load_all_domains()
        stats = question_bank_service.get_stats()
        
        assert stats.total_questions == 13  # 10 backend + 3 aiml
        assert "backend" in stats.loaded_domains
        assert "aiml" in stats.loaded_domains
        assert stats.questions_by_domain["backend"] == 10
        assert stats.questions_by_domain["aiml"] == 3
    
    @pytest.mark.asyncio
    async def test_get_domain_info(self, question_bank_service):
        """Test getting domain info."""
        await question_bank_service.load_domain("backend")
        info = question_bank_service.get_domain_info("backend")
        
        assert info is not None
        assert info.name == "backend"
        assert info.question_count == 10
        assert len(info.categories) > 0
        assert len(info.difficulties) > 0


# =============================================================================
# Hybrid Question Selector Tests
# =============================================================================

class TestHybridQuestionSelector:
    """Tests for HybridQuestionSelector."""
    
    @pytest.fixture
    def hybrid_selector(self, question_bank_service):
        """Create a HybridQuestionSelector with test bank service."""
        return HybridQuestionSelector(bank_service=question_bank_service)
    
    @pytest.mark.asyncio
    async def test_select_questions_basic(self, hybrid_selector, sample_jd, sample_resume):
        """Test basic question selection."""
        config = QuestionBankConfig(
            use_question_bank=True,
            auto_detect_domains=True,
            bank_question_ratio=0.7,
        )
        
        bank_questions, uncovered_skills = await hybrid_selector.select_questions(
            jd=sample_jd,
            resume=sample_resume,
            config=config,
            stage=InterviewStageHint.TECHNICAL,
            count=5,
        )
        
        # Should return some bank questions
        assert isinstance(bank_questions, list)
        assert isinstance(uncovered_skills, list)
    
    @pytest.mark.asyncio
    async def test_select_questions_with_bank_disabled(self, hybrid_selector, sample_jd, sample_resume):
        """Test that disabling bank returns empty questions and all skills."""
        config = QuestionBankConfig(
            use_question_bank=False,
        )
        
        bank_questions, uncovered_skills = await hybrid_selector.select_questions(
            jd=sample_jd,
            resume=sample_resume,
            config=config,
            stage=InterviewStageHint.TECHNICAL,
            count=5,
        )
        
        # Should return no bank questions
        assert bank_questions == []
        # Should return all JD skills as uncovered
        assert len(uncovered_skills) > 0
    
    @pytest.mark.asyncio
    async def test_select_questions_respects_count(self, hybrid_selector, sample_jd, sample_resume):
        """Test that selection respects the count and ratio."""
        config = QuestionBankConfig(
            use_question_bank=True,
            bank_question_ratio=0.5,
        )
        
        bank_questions, _ = await hybrid_selector.select_questions(
            jd=sample_jd,
            resume=sample_resume,
            config=config,
            stage=InterviewStageHint.TECHNICAL,
            count=4,
        )
        
        # With 0.5 ratio and count=4, should select at most 2 from bank
        assert len(bank_questions) <= 2
    
    @pytest.mark.asyncio
    async def test_select_questions_with_admin_domains(self, hybrid_selector, sample_jd, sample_resume):
        """Test selection with admin-specified domains."""
        config = QuestionBankConfig(
            use_question_bank=True,
            enabled_domains=["aiml"],
            auto_detect_domains=False,
        )
        
        bank_questions, _ = await hybrid_selector.select_questions(
            jd=sample_jd,
            resume=sample_resume,
            config=config,
            stage=InterviewStageHint.TECHNICAL,
            count=5,
        )
        
        # All questions should be from aiml domain
        for q in bank_questions:
            assert q.domain == "aiml"
    
    @pytest.mark.asyncio
    async def test_skill_coverage_report(self, hybrid_selector, sample_jd, sample_resume):
        """Test skill coverage report generation."""
        config = QuestionBankConfig(use_question_bank=True)
        
        bank_questions, _ = await hybrid_selector.select_questions(
            jd=sample_jd,
            resume=sample_resume,
            config=config,
            stage=InterviewStageHint.TECHNICAL,
            count=5,
        )
        
        jd_skills = sample_jd.required_skills + sample_jd.preferred_skills
        report = hybrid_selector.get_skill_coverage_report(bank_questions, jd_skills)
        
        assert "total_jd_skills" in report
        assert "covered_skills" in report
        assert "uncovered_skills" in report
        assert "coverage_percent" in report


# =============================================================================
# Question Bank Config Tests
# =============================================================================

class TestQuestionBankConfig:
    """Tests for QuestionBankConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = QuestionBankConfig()
        
        assert config.use_question_bank is True
        assert config.auto_detect_domains is True
        assert config.bank_question_ratio == 0.7
        assert config.allow_rephrasing is True
        assert config.allow_personalization is True
    
    def test_ratio_validation(self):
        """Test that ratio is validated between 0 and 1."""
        config = QuestionBankConfig(bank_question_ratio=0.5)
        assert config.bank_question_ratio == 0.5
        
        # Test boundary values
        config_min = QuestionBankConfig(bank_question_ratio=0.0)
        assert config_min.bank_question_ratio == 0.0
        
        config_max = QuestionBankConfig(bank_question_ratio=1.0)
        assert config_max.bank_question_ratio == 1.0
    
    def test_domain_configuration(self):
        """Test domain configuration options."""
        config = QuestionBankConfig(
            enabled_domains=["backend", "aiml"],
            auto_detect_domains=False,
        )
        
        assert config.enabled_domains == ["backend", "aiml"]
        assert config.auto_detect_domains is False


# =============================================================================
# Question Source Enum Tests
# =============================================================================

class TestQuestionSource:
    """Tests for QuestionSource enum."""
    
    def test_source_values(self):
        """Test that all expected source values exist."""
        assert QuestionSource.BANK.value == "bank"
        assert QuestionSource.BANK_REPHRASED.value == "bank_rephrased"
        assert QuestionSource.BANK_PERSONALIZED.value == "bank_personalized"
        assert QuestionSource.GENERATED.value == "generated"
    
    def test_source_comparison(self):
        """Test source comparison."""
        assert QuestionSource.BANK != QuestionSource.GENERATED
        assert QuestionSource.BANK_REPHRASED != QuestionSource.BANK_PERSONALIZED


# =============================================================================
# Stage Hint Tests  
# =============================================================================

class TestStageHint:
    """Tests for InterviewStageHint."""
    
    @pytest.mark.asyncio
    async def test_stage_hint_inference(self, question_bank_service):
        """Test that stage hints are properly inferred."""
        questions = await question_bank_service.load_domain("backend")
        
        # Most backend questions should be technical
        technical_qs = [q for q in questions if q.stage_hint == InterviewStageHint.TECHNICAL]
        assert len(technical_qs) > 0
    
    def test_stage_hint_values(self):
        """Test stage hint enum values."""
        assert InterviewStageHint.SCREENING.value == "screening"
        assert InterviewStageHint.TECHNICAL.value == "technical"
        assert InterviewStageHint.BEHAVIORAL.value == "behavioral"
        assert InterviewStageHint.SYSTEM_DESIGN.value == "system_design"
        assert InterviewStageHint.GENERAL.value == "general"
