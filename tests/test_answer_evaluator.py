"""
Tests for the Answer Evaluator service (LLM-as-Judge).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.services.answer_evaluator import (
    AnswerEvaluator,
    AnswerEvaluation,
    EvaluationScores,
    STARScores,
    InterviewReport,
    Recommendation,
    AnswerStrength,
    get_answer_evaluator,
)
from src.services.prompts import InterviewStage
from src.providers.llm import LLMResponse, GenerationConfig


def create_mock_evaluation_response(scores: dict = None, **kwargs) -> LLMResponse:
    """Create a mock LLM response for evaluation."""
    default_scores = {
        "technical_accuracy": 80,
        "completeness": 75,
        "clarity": 85,
        "depth": 70,
        "overall": 78,
    }
    response_data = {
        "scores": scores or default_scores,
        "strengths": kwargs.get("strengths", ["Good explanation", "Clear structure"]),
        "improvements": kwargs.get("improvements", ["Could add more examples"]),
        "follow_up_question": kwargs.get("follow_up_question", "Can you provide a specific example?"),
        "recommendation": kwargs.get("recommendation", "acceptable"),
        "notes": kwargs.get("notes", "Solid answer overall."),
    }
    return LLMResponse(
        content=json.dumps(response_data),
        model="mock-model",
        usage={"prompt_tokens": 75, "completion_tokens": 75, "total_tokens": 150},
        finish_reason="stop",
    )


def create_mock_behavioral_response(**kwargs) -> LLMResponse:
    """Create a mock LLM response for behavioral evaluation."""
    response_data = {
        "star_scores": {
            "situation": kwargs.get("situation", 80),
            "task": kwargs.get("task", 75),
            "action": kwargs.get("action", 85),
            "result": kwargs.get("result", 70),
            "total": kwargs.get("total", 78),
        },
        "overall_score": kwargs.get("overall_score", 78),
        "self_awareness_score": kwargs.get("self_awareness_score", 80),
        "relevance_score": kwargs.get("relevance_score", 75),
        "authenticity_score": kwargs.get("authenticity_score", 85),
        "green_flags_detected": kwargs.get("green_flags", ["Takes ownership", "Specific details"]),
        "red_flags_detected": kwargs.get("red_flags", []),
        "recommendation": kwargs.get("recommendation", "strong"),
        "notes": kwargs.get("notes", "Excellent behavioral response."),
    }
    return LLMResponse(
        content=json.dumps(response_data),
        model="mock-model",
        usage={"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        finish_reason="stop",
    )


def create_mock_validation_response(is_grounded: bool = True, issues: list = None) -> LLMResponse:
    """Create a mock LLM response for hallucination check."""
    response_data = {
        "is_grounded": is_grounded,
        "issues": issues or [],
        "confidence": 90 if is_grounded else 40,
    }
    return LLMResponse(
        content=json.dumps(response_data),
        model="mock-model",
        usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        finish_reason="stop",
    )


class TestEvaluationScores:
    """Tests for EvaluationScores dataclass."""
    
    def test_to_dict(self):
        """Test converting scores to dictionary."""
        scores = EvaluationScores(
            technical_accuracy=80,
            completeness=75,
            clarity=85,
            depth=70,
            overall=78,
        )
        
        result = scores.to_dict()
        
        assert result["technical_accuracy"] == 80
        assert result["completeness"] == 75
        assert result["overall"] == 78
    
    def test_from_dict(self):
        """Test creating scores from dictionary."""
        data = {
            "technical_accuracy": 80,
            "completeness": 75,
            "clarity": 85,
            "depth": 70,
            "overall": 78,
        }
        
        scores = EvaluationScores.from_dict(data)
        
        assert scores.technical_accuracy == 80
        assert scores.overall == 78
    
    def test_from_dict_with_missing_fields(self):
        """Test creating scores from incomplete dictionary."""
        data = {"overall": 70}
        
        scores = EvaluationScores.from_dict(data)
        
        assert scores.overall == 70
        assert scores.technical_accuracy == 0


class TestSTARScores:
    """Tests for STARScores dataclass."""
    
    def test_to_dict(self):
        """Test converting STAR scores to dictionary."""
        scores = STARScores(
            situation=80,
            task=75,
            action=85,
            result=70,
            total=78,
        )
        
        result = scores.to_dict()
        
        assert result["situation"] == 80
        assert result["action"] == 85
        assert result["total"] == 78
    
    def test_from_dict(self):
        """Test creating STAR scores from dictionary."""
        data = {
            "situation": 80,
            "task": 75,
            "action": 85,
            "result": 70,
            "total": 78,
        }
        
        scores = STARScores.from_dict(data)
        
        assert scores.situation == 80
        assert scores.total == 78


class TestAnswerEvaluation:
    """Tests for AnswerEvaluation dataclass."""
    
    def test_to_dict(self):
        """Test converting evaluation to dictionary."""
        evaluation = AnswerEvaluation(
            question="What is Python?",
            answer="Python is a programming language.",
            stage=InterviewStage.TECHNICAL,
            scores=EvaluationScores(overall=75),
            strengths=["Clear explanation"],
            improvements=["Add more depth"],
            recommendation=AnswerStrength.ACCEPTABLE,
        )
        
        result = evaluation.to_dict()
        
        assert result["question"] == "What is Python?"
        assert result["stage"] == "technical"
        assert result["recommendation"] == "acceptable"
        assert len(result["strengths"]) == 1
    
    def test_to_dict_with_star_scores(self):
        """Test evaluation with STAR scores."""
        evaluation = AnswerEvaluation(
            question="Tell me about a challenge.",
            answer="At my previous job...",
            stage=InterviewStage.BEHAVIORAL,
            scores=EvaluationScores(overall=80),
            star_scores=STARScores(situation=85, task=80, action=90, result=75, total=82),
        )
        
        result = evaluation.to_dict()
        
        assert result["star_scores"] is not None
        assert result["star_scores"]["situation"] == 85


class TestInterviewReport:
    """Tests for InterviewReport dataclass."""
    
    def test_to_dict(self):
        """Test converting report to dictionary."""
        evaluation = AnswerEvaluation(
            question="Test question",
            answer="Test answer",
            stage=InterviewStage.TECHNICAL,
            scores=EvaluationScores(overall=75),
        )
        
        report = InterviewReport(
            candidate_name="John Doe",
            role_title="Software Engineer",
            duration_minutes=45,
            evaluations=[evaluation],
            overall_score=75,
            recommendation=Recommendation.HIRE,
        )
        
        result = report.to_dict()
        
        assert result["candidate_name"] == "John Doe"
        assert result["recommendation"] == "hire"
        assert len(result["evaluations"]) == 1


class TestAnswerEvaluator:
    """Tests for AnswerEvaluator class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock
    
    @pytest.fixture
    def evaluator(self, mock_llm):
        """Create an AnswerEvaluator with mock LLM."""
        return AnswerEvaluator(llm_provider=mock_llm)
    
    def test_initialization_with_custom_llm(self, mock_llm):
        """Test that evaluator initializes with custom LLM provider."""
        evaluator = AnswerEvaluator(llm_provider=mock_llm)
        assert evaluator.llm is mock_llm
    
    def test_parse_json_response_valid(self, evaluator):
        """Test parsing valid JSON object."""
        response = '{"scores": {"overall": 75}, "strengths": []}'
        result = evaluator._parse_json_response(response)
        
        assert result["scores"]["overall"] == 75
    
    def test_parse_json_response_with_markdown(self, evaluator):
        """Test parsing JSON wrapped in markdown."""
        response = '```json\n{"scores": {"overall": 75}}\n```'
        result = evaluator._parse_json_response(response)
        
        assert result["scores"]["overall"] == 75
    
    def test_parse_json_response_with_extra_text(self, evaluator):
        """Test parsing JSON with surrounding text."""
        response = 'Here is the evaluation:\n{"scores": {"overall": 75}}\nEnd.'
        result = evaluator._parse_json_response(response)
        
        assert result["scores"]["overall"] == 75
    
    def test_parse_json_response_invalid(self, evaluator):
        """Test parsing invalid JSON returns empty dict."""
        response = 'This is not valid JSON'
        result = evaluator._parse_json_response(response)
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_technical(self, evaluator, mock_llm):
        """Test evaluating a technical answer."""
        mock_llm.generate.side_effect = [
            create_mock_evaluation_response(),
            create_mock_validation_response(is_grounded=True),
        ]
        
        evaluation = await evaluator.evaluate_answer(
            question="Explain REST APIs.",
            answer="REST APIs use HTTP methods to interact with resources...",
            expected_points=["HTTP methods", "Resources", "Stateless"],
            stage=InterviewStage.TECHNICAL,
            validate=True,
        )
        
        assert evaluation.question == "Explain REST APIs."
        assert evaluation.stage == InterviewStage.TECHNICAL
        assert evaluation.scores.overall > 0
        assert evaluation.is_validated is True
        assert len(evaluation.strengths) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_without_validation(self, evaluator, mock_llm):
        """Test evaluating without hallucination check."""
        mock_llm.generate.return_value = create_mock_evaluation_response()
        
        evaluation = await evaluator.evaluate_answer(
            question="What is Python?",
            answer="Python is a programming language.",
            expected_points=["Interpreted", "High-level"],
            validate=False,
        )
        
        assert evaluation.is_validated is True
        # Should only call generate once (no validation call)
        assert mock_llm.generate.call_count == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_validation_fails(self, evaluator, mock_llm):
        """Test handling when validation finds issues."""
        mock_llm.generate.side_effect = [
            create_mock_evaluation_response(),
            create_mock_validation_response(
                is_grounded=False,
                issues=[{"description": "Score seems too high for brief answer"}]
            ),
        ]
        
        evaluation = await evaluator.evaluate_answer(
            question="Explain microservices.",
            answer="They are small services.",
            expected_points=["Independent deployment", "Loose coupling"],
            validate=True,
        )
        
        assert evaluation.is_validated is False
        assert len(evaluation.validation_issues) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_llm_error(self, evaluator, mock_llm):
        """Test error handling when LLM fails."""
        mock_llm.generate.side_effect = Exception("LLM connection error")
        
        evaluation = await evaluator.evaluate_answer(
            question="Test question",
            answer="Test answer",
            expected_points=[],
        )
        
        assert evaluation.scores.overall == 50  # Default score
        assert "error" in evaluation.notes.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_parse_failure(self, evaluator, mock_llm):
        """Test handling when JSON parsing fails."""
        mock_llm.generate.return_value = LLMResponse(
            content="This is not valid JSON response",
            model="mock-model",
            usage={"prompt_tokens": 25, "completion_tokens": 25, "total_tokens": 50},
            finish_reason="stop",
        )
        
        evaluation = await evaluator.evaluate_answer(
            question="Test question",
            answer="Test answer",
            expected_points=[],
            validate=False,
        )
        
        assert evaluation.scores.overall == 50
        assert "parsing failed" in evaluation.notes.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_behavioral_answer(self, evaluator, mock_llm):
        """Test evaluating a behavioral answer."""
        mock_llm.generate.side_effect = [
            create_mock_behavioral_response(),
            create_mock_validation_response(is_grounded=True),
        ]
        
        evaluation = await evaluator.evaluate_behavioral_answer(
            question="Tell me about a time you led a team.",
            answer="In my previous role, I led a team of 5 developers...",
            competency="leadership",
            red_flags=["Blaming others", "Vague answers"],
            green_flags=["Takes ownership", "Specific details"],
            validate=True,
        )
        
        assert evaluation.stage == InterviewStage.BEHAVIORAL
        assert evaluation.star_scores is not None
        assert evaluation.star_scores.situation > 0
        assert evaluation.is_validated is True
    
    @pytest.mark.asyncio
    async def test_evaluate_behavioral_without_validation(self, evaluator, mock_llm):
        """Test behavioral evaluation without validation."""
        mock_llm.generate.return_value = create_mock_behavioral_response()
        
        evaluation = await evaluator.evaluate_behavioral_answer(
            question="Describe a conflict you resolved.",
            answer="There was a disagreement about architecture...",
            competency="conflict-resolution",
            red_flags=[],
            green_flags=[],
            validate=False,
        )
        
        assert evaluation.is_validated is True
        assert mock_llm.generate.call_count == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_behavioral_llm_error(self, evaluator, mock_llm):
        """Test error handling in behavioral evaluation."""
        mock_llm.generate.side_effect = Exception("LLM error")
        
        evaluation = await evaluator.evaluate_behavioral_answer(
            question="Test question",
            answer="Test answer",
            competency="teamwork",
            red_flags=[],
            green_flags=[],
        )
        
        assert evaluation.scores.overall == 50
        assert "error" in evaluation.notes.lower()
    
    @pytest.mark.asyncio
    async def test_validate_evaluation(self, evaluator, mock_llm):
        """Test the validation helper method."""
        mock_llm.generate.return_value = create_mock_validation_response(is_grounded=True)
        
        result = await evaluator._validate_evaluation(
            question="What is REST?",
            answer="REST is an architectural style...",
            evaluation={"scores": {"overall": 80}},
        )
        
        assert result["is_grounded"] is True
        assert result.get("confidence", 0) > 0
    
    @pytest.mark.asyncio
    async def test_validate_evaluation_error(self, evaluator, mock_llm):
        """Test validation returns grounded=True on error."""
        mock_llm.generate.side_effect = Exception("Validation error")
        
        result = await evaluator._validate_evaluation(
            question="Test",
            answer="Test",
            evaluation={},
        )
        
        # On error, should return grounded=True to avoid false negatives
        assert result["is_grounded"] is True
    
    @pytest.mark.asyncio
    async def test_generate_interview_report(self, evaluator, mock_llm):
        """Test generating a complete interview report."""
        mock_llm.generate.return_value = LLMResponse(
            content=json.dumps({
                "executive_summary": "Strong candidate with solid technical skills.",
                "strengths": [{"area": "Technical", "detail": "Deep Python knowledge"}],
                "concerns": [{"area": "Experience", "detail": "Limited cloud experience"}],
                "recommendation": "hire",
                "confidence": 85,
                "reasoning": "Technical skills outweigh minor gaps.",
                "next_steps": ["Technical deep-dive", "Team fit interview"],
            }),
            model="mock-model",
            usage={"prompt_tokens": 150, "completion_tokens": 150, "total_tokens": 300},
            finish_reason="stop",
        )
        
        evaluations = [
            AnswerEvaluation(
                question="Python question",
                answer="Python answer",
                stage=InterviewStage.TECHNICAL,
                scores=EvaluationScores(overall=80, clarity=85, depth=75),
            ),
            AnswerEvaluation(
                question="Behavioral question",
                answer="Behavioral answer",
                stage=InterviewStage.BEHAVIORAL,
                scores=EvaluationScores(overall=75, clarity=80),
            ),
        ]
        
        report = await evaluator.generate_interview_report(
            candidate_name="John Doe",
            role_title="Senior Python Developer",
            duration_minutes=45,
            evaluations=evaluations,
        )
        
        assert report.candidate_name == "John Doe"
        assert report.role_title == "Senior Python Developer"
        assert report.technical_score > 0
        assert report.behavioral_score > 0
        assert report.recommendation == Recommendation.HIRE
        assert len(report.next_steps) > 0
    
    @pytest.mark.asyncio
    async def test_generate_interview_report_empty_evaluations(self, evaluator, mock_llm):
        """Test report generation with no evaluations."""
        mock_llm.generate.return_value = LLMResponse(
            content=json.dumps({
                "executive_summary": "No responses to evaluate.",
                "recommendation": "no_hire",
                "confidence": 20,
            }),
            model="mock-model",
            usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
            finish_reason="stop",
        )
        
        report = await evaluator.generate_interview_report(
            candidate_name="Jane Smith",
            role_title="Developer",
            duration_minutes=30,
            evaluations=[],
        )
        
        assert report.technical_score == 0
        assert report.behavioral_score == 0
    
    @pytest.mark.asyncio
    async def test_generate_interview_report_llm_error(self, evaluator, mock_llm):
        """Test report generation handles LLM errors."""
        mock_llm.generate.side_effect = Exception("LLM error")
        
        evaluations = [
            AnswerEvaluation(
                question="Test",
                answer="Test",
                stage=InterviewStage.TECHNICAL,
                scores=EvaluationScores(overall=80),
            ),
        ]
        
        report = await evaluator.generate_interview_report(
            candidate_name="Test Candidate",
            role_title="Developer",
            duration_minutes=30,
            evaluations=evaluations,
        )
        
        # Should still return report with calculated scores
        assert report.technical_score == 80
        assert "80" in report.executive_summary or "completed" in report.executive_summary.lower()


class TestAnswerEvaluatorSingleton:
    """Tests for the singleton pattern."""
    
    def test_get_answer_evaluator_returns_same_instance(self):
        """Test that get_answer_evaluator returns singleton."""
        # Reset global state
        import src.services.answer_evaluator as ae
        ae._answer_evaluator = None
        
        with patch('src.services.answer_evaluator.get_llm_provider_sync') as mock_provider:
            mock_provider.return_value = MagicMock()
            
            eval1 = get_answer_evaluator()
            eval2 = get_answer_evaluator()
            
            assert eval1 is eval2
        
        # Cleanup
        ae._answer_evaluator = None


class TestRecommendationEnum:
    """Tests for the Recommendation enum."""
    
    def test_recommendation_values(self):
        """Test that all recommendation values are correct."""
        assert Recommendation.STRONG_HIRE.value == "strong_hire"
        assert Recommendation.HIRE.value == "hire"
        assert Recommendation.NO_HIRE.value == "no_hire"
        assert Recommendation.STRONG_NO_HIRE.value == "strong_no_hire"


class TestAnswerStrengthEnum:
    """Tests for the AnswerStrength enum."""
    
    def test_answer_strength_values(self):
        """Test that all strength values are correct."""
        assert AnswerStrength.STRONG.value == "strong"
        assert AnswerStrength.ACCEPTABLE.value == "acceptable"
        assert AnswerStrength.WEAK.value == "weak"
        assert AnswerStrength.INSUFFICIENT.value == "insufficient"


# Run with: pytest tests/test_answer_evaluator.py -v
