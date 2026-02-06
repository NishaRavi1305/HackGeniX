"""
Tests for the Question Generator service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.services.question_generator import (
    QuestionGenerator,
    QuestionGenerationRequest,
    GeneratedQuestion,
    get_question_generator,
)
from src.services.prompts import InterviewStage, QuestionDifficulty
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo, Experience
from src.providers.llm import LLMResponse, GenerationConfig


# Sample data fixtures
def create_sample_resume() -> ParsedResume:
    """Create a sample parsed resume for testing."""
    return ParsedResume(
        contact=ContactInfo(
            name="John Doe",
            email="john@example.com",
        ),
        summary="Experienced Python developer with 5 years of experience in web development and machine learning.",
        skills=["python", "javascript", "react", "django", "fastapi", "postgresql", "aws", "docker"],
        experience=[
            Experience(
                company="TechCorp",
                title="Senior Software Engineer",
                description="Led development of microservices using Python and FastAPI",
                highlights=["Built ML pipeline", "Reduced latency by 40%"],
            ),
            Experience(
                company="StartupXYZ",
                title="Software Engineer",
                description="Full-stack development with React and Django",
            ),
        ],
        raw_text="John Doe - Python Developer with experience in web and ML...",
    )


def create_sample_jd() -> ParsedJobDescription:
    """Create a sample parsed job description for testing."""
    return ParsedJobDescription(
        title="Senior Python Developer",
        company="TechStartup",
        required_skills=["python", "django", "postgresql", "aws", "docker", "kubernetes"],
        preferred_skills=["react", "fastapi", "machine learning"],
        responsibilities=[
            "Design and implement scalable backend services",
            "Build RESTful APIs",
            "Mentor junior developers",
        ],
        experience_years_min=5,
        raw_text="We are looking for a Senior Python Developer...",
    )


def create_mock_llm_response(questions: list) -> LLMResponse:
    """Create a mock LLM response with questions."""
    return LLMResponse(
        content=json.dumps(questions),
        model="mock-model",
        usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        finish_reason="stop",
    )


class TestGeneratedQuestion:
    """Tests for GeneratedQuestion dataclass."""
    
    def test_to_dict(self):
        """Test converting GeneratedQuestion to dictionary."""
        question = GeneratedQuestion(
            question="What is a REST API?",
            stage=InterviewStage.TECHNICAL,
            difficulty=QuestionDifficulty.MEDIUM,
            category="API Design",
            purpose="Assess API knowledge",
            expected_answer_points=["HTTP methods", "Resources", "Stateless"],
            follow_up_questions=["How do you version APIs?"],
            duration_seconds=180,
        )
        
        result = question.to_dict()
        
        assert result["question"] == "What is a REST API?"
        assert result["stage"] == "technical"
        assert result["difficulty"] == "medium"
        assert result["category"] == "API Design"
        assert len(result["expected_answer_points"]) == 3


class TestQuestionGenerationRequest:
    """Tests for QuestionGenerationRequest dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        request = QuestionGenerationRequest(stage=InterviewStage.TECHNICAL)
        
        assert request.stage == InterviewStage.TECHNICAL
        assert request.num_questions == 5
        assert request.difficulty == QuestionDifficulty.MEDIUM
        assert request.focus_areas == []
        assert request.exclude_topics == []
    
    def test_custom_values(self):
        """Test custom values are preserved."""
        request = QuestionGenerationRequest(
            stage=InterviewStage.BEHAVIORAL,
            num_questions=3,
            difficulty=QuestionDifficulty.HARD,
            focus_areas=["leadership", "teamwork"],
        )
        
        assert request.num_questions == 3
        assert request.difficulty == QuestionDifficulty.HARD
        assert "leadership" in request.focus_areas


class TestQuestionGenerator:
    """Tests for QuestionGenerator class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock
    
    @pytest.fixture
    def generator(self, mock_llm):
        """Create a QuestionGenerator with mock LLM."""
        return QuestionGenerator(llm_provider=mock_llm)
    
    def test_initialization_with_custom_llm(self, mock_llm):
        """Test that generator initializes with custom LLM provider."""
        generator = QuestionGenerator(llm_provider=mock_llm)
        assert generator.llm is mock_llm
    
    def test_build_resume_summary(self, generator):
        """Test resume summary building."""
        resume = create_sample_resume()
        summary = generator._build_resume_summary(resume)
        
        assert "Python developer" in summary or "python" in summary.lower()
        assert "TechCorp" in summary or "Senior Software Engineer" in summary
    
    def test_build_resume_summary_empty(self, generator):
        """Test resume summary with minimal data."""
        resume = ParsedResume(raw_text="")
        summary = generator._build_resume_summary(resume)
        
        assert summary == "No resume details available"
    
    def test_build_jd_summary(self, generator):
        """Test job description summary building."""
        jd = create_sample_jd()
        summary = generator._build_jd_summary(jd)
        
        assert "Senior Python Developer" in summary
        assert "python" in summary.lower()
    
    def test_build_jd_summary_empty(self, generator):
        """Test JD summary with minimal data."""
        jd = ParsedJobDescription(raw_text="")
        summary = generator._build_jd_summary(jd)
        
        assert summary == "No JD details available"
    
    def test_parse_llm_json_response_valid_json(self, generator):
        """Test parsing valid JSON array."""
        response = '[{"question": "What is Python?"}]'
        result = generator._parse_llm_json_response(response)
        
        assert len(result) == 1
        assert result[0]["question"] == "What is Python?"
    
    def test_parse_llm_json_response_with_markdown(self, generator):
        """Test parsing JSON wrapped in markdown code block."""
        response = '```json\n[{"question": "What is Python?"}]\n```'
        result = generator._parse_llm_json_response(response)
        
        assert len(result) == 1
        assert result[0]["question"] == "What is Python?"
    
    def test_parse_llm_json_response_with_extra_text(self, generator):
        """Test parsing JSON with surrounding text."""
        response = 'Here are the questions:\n[{"question": "What is Python?"}]\nEnd of questions.'
        result = generator._parse_llm_json_response(response)
        
        assert len(result) == 1
        assert result[0]["question"] == "What is Python?"
    
    def test_parse_llm_json_response_invalid_json(self, generator):
        """Test parsing invalid JSON returns empty list."""
        response = 'This is not valid JSON at all'
        result = generator._parse_llm_json_response(response)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_generate_questions_technical(self, generator, mock_llm):
        """Test generating technical questions."""
        mock_questions = [
            {
                "question": "Explain Python decorators and when you would use them.",
                "difficulty": "medium",
                "category": "Python",
                "purpose": "Assess Python knowledge",
                "expected_answer_points": ["Function wrapper", "Syntax @", "Use cases"],
                "follow_up_questions": ["Can you write a simple decorator?"],
                "duration_seconds": 180,
            },
            {
                "question": "What is the difference between a list and a tuple in Python?",
                "difficulty": "easy",
                "category": "Python",
                "purpose": "Assess basic Python knowledge",
                "expected_answer_points": ["Mutability", "Syntax", "Performance"],
                "follow_up_questions": [],
                "duration_seconds": 120,
            },
        ]
        mock_llm.generate.return_value = create_mock_llm_response(mock_questions)
        
        request = QuestionGenerationRequest(
            stage=InterviewStage.TECHNICAL,
            num_questions=2,
        )
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        questions = await generator.generate_questions(request, resume, jd)
        
        assert len(questions) == 2
        assert questions[0].stage == InterviewStage.TECHNICAL
        assert "decorator" in questions[0].question.lower() or "python" in questions[0].question.lower()
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_previous(self, generator, mock_llm):
        """Test that previous questions are included in prompt."""
        mock_llm.generate.return_value = create_mock_llm_response([
            {"question": "New question here", "difficulty": "medium"}
        ])
        
        request = QuestionGenerationRequest(
            stage=InterviewStage.TECHNICAL,
            num_questions=1,
        )
        resume = create_sample_resume()
        jd = create_sample_jd()
        previous = ["What is Python?", "Explain OOP."]
        
        await generator.generate_questions(request, resume, jd, previous_questions=previous)
        
        # Verify the prompt includes previous questions
        call_args = mock_llm.generate.call_args
        messages = call_args[0][0]  # First positional argument is messages
        prompt_text = messages[-1].content  # Last message is user prompt
        assert "What is Python?" in prompt_text or "already asked" in prompt_text.lower()
    
    @pytest.mark.asyncio
    async def test_generate_questions_invalid_stage(self, generator, mock_llm):
        """Test error handling for invalid stage."""
        # Create a request with a valid stage first, then mock the check
        request = QuestionGenerationRequest(
            stage=InterviewStage.TECHNICAL,
            num_questions=1,
        )
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        # Force an invalid stage scenario by manipulating the request
        request.stage = "invalid_stage"  # type: ignore
        
        with pytest.raises((ValueError, KeyError)):
            await generator.generate_questions(request, resume, jd)
    
    @pytest.mark.asyncio
    async def test_generate_follow_up(self, generator, mock_llm):
        """Test generating follow-up questions."""
        mock_llm.generate.return_value = LLMResponse(
            content='{"follow_up_question": "Can you elaborate on the error handling?"}',
            model="mock-model",
            usage={"prompt_tokens": 25, "completion_tokens": 25, "total_tokens": 50},
            finish_reason="stop",
        )
        
        follow_up = await generator.generate_follow_up(
            original_question="How do you handle errors in Python?",
            candidate_answer="I use try-except blocks.",
            evaluation_summary="Answer is too brief.",
        )
        
        assert follow_up is not None
        assert "elaborate" in follow_up.lower() or "error" in follow_up.lower()
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_returns_none_on_error(self, generator, mock_llm):
        """Test that follow-up returns None on error."""
        mock_llm.generate.side_effect = Exception("LLM error")
        
        follow_up = await generator.generate_follow_up(
            original_question="Test question",
            candidate_answer="Test answer",
            evaluation_summary="Test summary",
        )
        
        assert follow_up is None
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty(self, generator, mock_llm):
        """Test difficulty adjustment based on performance."""
        mock_llm.generate.return_value = LLMResponse(
            content='{"next_difficulty": "hard", "difficulty_change": "increase", "reasoning": "Strong performance"}',
            model="mock-model",
            usage={"prompt_tokens": 25, "completion_tokens": 25, "total_tokens": 50},
            finish_reason="stop",
        )
        
        performance = {
            "questions_answered": 5,
            "average_score": 85,
            "trend": "improving",
            "strong_areas": ["algorithms"],
            "weak_areas": [],
            "current_difficulty": "medium",
        }
        
        result = await generator.adjust_difficulty(performance)
        
        assert result.get("next_difficulty") == "hard"
        assert result.get("difficulty_change") == "increase"
    
    @pytest.mark.asyncio
    async def test_generate_screening_questions(self, generator, mock_llm):
        """Test convenience method for screening questions."""
        mock_llm.generate.return_value = create_mock_llm_response([
            {"question": "Walk me through your background.", "difficulty": "easy"}
        ])
        
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        questions = await generator.generate_screening_questions(resume, jd, num_questions=1)
        
        assert len(questions) == 1
        assert questions[0].stage == InterviewStage.SCREENING
    
    @pytest.mark.asyncio
    async def test_generate_technical_questions(self, generator, mock_llm):
        """Test convenience method for technical questions."""
        mock_llm.generate.return_value = create_mock_llm_response([
            {"question": "Explain REST APIs.", "difficulty": "medium"}
        ])
        
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        questions = await generator.generate_technical_questions(
            resume, jd, num_questions=1, focus_areas=["APIs"]
        )
        
        assert len(questions) == 1
        assert questions[0].stage == InterviewStage.TECHNICAL
    
    @pytest.mark.asyncio
    async def test_generate_behavioral_questions(self, generator, mock_llm):
        """Test convenience method for behavioral questions."""
        mock_llm.generate.return_value = create_mock_llm_response([
            {"question": "Tell me about a challenging project.", "difficulty": "medium", "competency": "problem-solving"}
        ])
        
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        questions = await generator.generate_behavioral_questions(resume, jd, num_questions=1)
        
        assert len(questions) == 1
        assert questions[0].stage == InterviewStage.BEHAVIORAL


class TestQuestionGeneratorSingleton:
    """Tests for the singleton pattern."""
    
    def test_get_question_generator_returns_same_instance(self):
        """Test that get_question_generator returns singleton."""
        # Reset global state
        import src.services.question_generator as qg
        qg._question_generator = None
        
        with patch('src.services.question_generator.get_llm_provider_sync') as mock_provider:
            mock_provider.return_value = MagicMock()
            
            gen1 = get_question_generator()
            gen2 = get_question_generator()
            
            assert gen1 is gen2
        
        # Cleanup
        qg._question_generator = None


# Run with: pytest tests/test_question_generator.py -v
