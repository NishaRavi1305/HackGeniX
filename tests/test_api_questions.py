"""
API Endpoint Integration Tests for Questions and Evaluation.

Tests the FastAPI endpoints with mocked services to verify:
- Request validation
- Response formats
- Error handling
- HTTP status codes
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId

from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app
from src.services.question_generator import GeneratedQuestion
from src.services.answer_evaluator import AnswerEvaluation, EvaluationScores, AnswerStrength
from src.services.prompts import InterviewStage, QuestionDifficulty


# API base path for questions router
API_PREFIX = "/api/v1/questions"


# Test data fixtures
@pytest.fixture
def valid_object_id():
    """Generate a valid MongoDB ObjectId."""
    return str(ObjectId())


@pytest.fixture
def sample_jd_doc(valid_object_id):
    """Sample job description document."""
    return {
        "_id": ObjectId(valid_object_id),
        "title": "Senior Backend Engineer",
        "company": "TechCorp",
        "raw_text": "Looking for a senior backend engineer...",
        "parsed_data": {
            "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            "preferred_skills": ["Kubernetes", "AWS"],
            "responsibilities": ["Design and build APIs", "Mentor junior devs"],
            "experience_years_min": 5,
        }
    }


@pytest.fixture
def sample_resume_doc(valid_object_id):
    """Sample resume document."""
    return {
        "_id": ObjectId(valid_object_id),
        "parsed_data": {
            "summary": "Experienced backend developer with 6 years experience",
            "skills": ["Python", "Django", "PostgreSQL", "Docker"],
            "raw_text": "John Doe - Software Engineer...",
        }
    }


@pytest.fixture
def sample_generated_questions():
    """Sample generated questions."""
    return [
        GeneratedQuestion(
            question="Explain how you would design a rate limiting system.",
            stage=InterviewStage.TECHNICAL,
            difficulty=QuestionDifficulty.MEDIUM,
            category="system-design",
            purpose="Test system design skills",
            expected_answer_points=["Token bucket", "Redis", "429 status"],
            follow_up_questions=["What about distributed systems?"],
            duration_seconds=180,
        ),
        GeneratedQuestion(
            question="How do you handle database migrations?",
            stage=InterviewStage.TECHNICAL,
            difficulty=QuestionDifficulty.EASY,
            purpose="Test practical knowledge",
            expected_answer_points=["Alembic", "Version control"],
        ),
    ]


@pytest.fixture
def sample_evaluation():
    """Sample answer evaluation."""
    return AnswerEvaluation(
        question="Explain rate limiting",
        answer="I would use token bucket algorithm...",
        stage=InterviewStage.TECHNICAL,
        scores=EvaluationScores(
            technical_accuracy=80,
            completeness=75,
            clarity=85,
            depth=70,
            overall=77.5,
        ),
        strengths=["Good algorithm knowledge", "Practical approach"],
        improvements=["Could discuss edge cases"],
        follow_up_question="How would you handle bursts?",
        recommendation=AnswerStrength.ACCEPTABLE,
        notes="Solid answer overall",
        is_validated=True,
        validation_issues=[],
    )


class TestGenerateQuestionsEndpoint:
    """Test POST /questions/generate endpoint."""
    
    @pytest.mark.asyncio
    async def test_generate_questions_success(
        self, valid_object_id, sample_jd_doc, sample_generated_questions
    ):
        """Test successful question generation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db, \
                 patch("src.api.questions.get_question_generator") as mock_gen:
                
                # Setup mocks
                mock_db.job_descriptions.find_one = AsyncMock(return_value=sample_jd_doc)
                mock_db.resumes.find_one = AsyncMock(return_value=None)
                
                mock_generator = MagicMock()
                mock_generator.generate_questions = AsyncMock(return_value=sample_generated_questions)
                mock_gen.return_value = mock_generator
                
                # Make request
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "stage": "technical",
                        "num_questions": 2,
                        "difficulty": "medium",
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "questions" in data
                assert len(data["questions"]) == 2
                assert data["stage"] == "technical"
                assert data["job_description_id"] == valid_object_id
                
                # Check question structure
                q = data["questions"][0]
                assert "question" in q
                assert "stage" in q
                assert "difficulty" in q
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_resume(
        self, valid_object_id, sample_jd_doc, sample_resume_doc, sample_generated_questions
    ):
        """Test question generation with resume context."""
        resume_id = str(ObjectId())
        sample_resume_doc["_id"] = ObjectId(resume_id)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db, \
                 patch("src.api.questions.get_question_generator") as mock_gen:
                
                mock_db.job_descriptions.find_one = AsyncMock(return_value=sample_jd_doc)
                mock_db.resumes.find_one = AsyncMock(return_value=sample_resume_doc)
                
                mock_generator = MagicMock()
                mock_generator.generate_questions = AsyncMock(return_value=sample_generated_questions)
                mock_gen.return_value = mock_generator
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "resume_id": resume_id,
                        "stage": "technical",
                        "num_questions": 2,
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["resume_id"] == resume_id
    
    @pytest.mark.asyncio
    async def test_generate_questions_invalid_jd_id(self):
        """Test with invalid job description ID format."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db:
                mock_db.job_descriptions.find_one = AsyncMock(side_effect=Exception("Invalid ObjectId"))
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": "invalid-id",
                        "stage": "technical",
                    }
                )
                
                assert response.status_code == 400
                assert "Invalid job description ID" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_questions_jd_not_found(self, valid_object_id):
        """Test when job description doesn't exist."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db:
                mock_db.job_descriptions.find_one = AsyncMock(return_value=None)
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "stage": "technical",
                    }
                )
                
                assert response.status_code == 404
                assert "Job description not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_questions_invalid_stage(self, valid_object_id, sample_jd_doc):
        """Test with invalid interview stage."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db:
                mock_db.job_descriptions.find_one = AsyncMock(return_value=sample_jd_doc)
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "stage": "invalid_stage",
                    }
                )
                
                assert response.status_code == 400
                assert "Invalid stage" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_questions_invalid_difficulty(self, valid_object_id, sample_jd_doc):
        """Test with invalid difficulty level."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db:
                mock_db.job_descriptions.find_one = AsyncMock(return_value=sample_jd_doc)
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "stage": "technical",
                        "difficulty": "impossible",
                    }
                )
                
                assert response.status_code == 400
                assert "Invalid difficulty" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_questions_llm_error(self, valid_object_id, sample_jd_doc):
        """Test handling of LLM generation errors."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.mongodb_client") as mock_db, \
                 patch("src.api.questions.get_question_generator") as mock_gen:
                
                mock_db.job_descriptions.find_one = AsyncMock(return_value=sample_jd_doc)
                
                mock_generator = MagicMock()
                mock_generator.generate_questions = AsyncMock(
                    side_effect=Exception("LLM connection failed")
                )
                mock_gen.return_value = mock_generator
                
                response = await client.post(
                    f"{API_PREFIX}/questions/generate",
                    json={
                        "job_description_id": valid_object_id,
                        "stage": "technical",
                    }
                )
                
                assert response.status_code == 500
                assert "Question generation failed" in response.json()["detail"]


class TestEvaluateAnswerEndpoint:
    """Test POST /answers/evaluate endpoint."""
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_success(self, sample_evaluation):
        """Test successful answer evaluation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_answer = AsyncMock(return_value=sample_evaluation)
                mock_eval.return_value = mock_evaluator
                
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate",
                    json={
                        "question": "Explain rate limiting",
                        "answer": "I would use token bucket...",
                        "expected_points": ["Token bucket", "Redis"],
                        "stage": "technical",
                        "validate": True,
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Check response structure
                assert data["question"] == "Explain rate limiting"
                assert data["stage"] == "technical"
                assert "scores" in data
                assert data["scores"]["overall"] == 77.5
                assert data["recommendation"] == "acceptable"
                assert data["is_validated"] == True
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_without_validation(self, sample_evaluation):
        """Test evaluation without hallucination check."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_answer = AsyncMock(return_value=sample_evaluation)
                mock_eval.return_value = mock_evaluator
                
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate",
                    json={
                        "question": "Test question",
                        "answer": "Test answer",
                        "validate": False,
                    }
                )
                
                assert response.status_code == 200
                # Verify validate=False was passed
                mock_evaluator.evaluate_answer.assert_called_once()
                call_kwargs = mock_evaluator.evaluate_answer.call_args.kwargs
                assert call_kwargs["validate"] == False
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_invalid_stage_defaults(self, sample_evaluation):
        """Test that invalid stage defaults to technical."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_answer = AsyncMock(return_value=sample_evaluation)
                mock_eval.return_value = mock_evaluator
                
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate",
                    json={
                        "question": "Test",
                        "answer": "Test",
                        "stage": "invalid_stage",
                    }
                )
                
                assert response.status_code == 200
                # Should default to TECHNICAL
                call_kwargs = mock_evaluator.evaluate_answer.call_args.kwargs
                assert call_kwargs["stage"] == InterviewStage.TECHNICAL
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_llm_error(self):
        """Test handling of evaluation errors."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_answer = AsyncMock(
                    side_effect=Exception("LLM timeout")
                )
                mock_eval.return_value = mock_evaluator
                
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate",
                    json={
                        "question": "Test",
                        "answer": "Test",
                    }
                )
                
                assert response.status_code == 500
                assert "Evaluation failed" in response.json()["detail"]


class TestEvaluateBehavioralEndpoint:
    """Test POST /answers/evaluate-behavioral endpoint."""
    
    @pytest.mark.asyncio
    async def test_evaluate_behavioral_success(self, sample_evaluation):
        """Test successful behavioral evaluation."""
        sample_evaluation.stage = InterviewStage.BEHAVIORAL
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_behavioral_answer = AsyncMock(return_value=sample_evaluation)
                mock_eval.return_value = mock_evaluator
                
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate-behavioral",
                    json={
                        "question": "Tell me about a conflict",
                        "answer": "In my previous role...",
                        "competency": "conflict-resolution",
                        "red_flags": ["Blaming others"],
                        "green_flags": ["Taking responsibility"],
                        "validate": True,
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["stage"] == "behavioral"
                assert "scores" in data
    
    @pytest.mark.asyncio
    async def test_evaluate_behavioral_with_defaults(self, sample_evaluation):
        """Test behavioral evaluation with default values."""
        sample_evaluation.stage = InterviewStage.BEHAVIORAL
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_answer_evaluator") as mock_eval:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_behavioral_answer = AsyncMock(return_value=sample_evaluation)
                mock_eval.return_value = mock_evaluator
                
                # Minimal request - only required fields
                response = await client.post(
                    f"{API_PREFIX}/answers/evaluate-behavioral",
                    json={
                        "question": "Tell me about a challenge",
                        "answer": "Once I had to...",
                    }
                )
                
                assert response.status_code == 200


class TestFollowUpEndpoint:
    """Test POST /questions/follow-up endpoint."""
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_success(self):
        """Test successful follow-up generation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_question_generator") as mock_gen:
                mock_generator = MagicMock()
                mock_generator.generate_follow_up = AsyncMock(
                    return_value="Can you elaborate on the caching strategy?"
                )
                mock_gen.return_value = mock_generator
                
                response = await client.post(
                    f"{API_PREFIX}/questions/follow-up",
                    json={
                        "original_question": "How would you design a cache?",
                        "candidate_answer": "I would use Redis...",
                        "evaluation_summary": "Good start but needs more depth",
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["follow_up_question"] == "Can you elaborate on the caching strategy?"
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_none(self):
        """Test when no follow-up is generated."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_question_generator") as mock_gen:
                mock_generator = MagicMock()
                mock_generator.generate_follow_up = AsyncMock(return_value=None)
                mock_gen.return_value = mock_generator
                
                response = await client.post(
                    f"{API_PREFIX}/questions/follow-up",
                    json={
                        "original_question": "Test",
                        "candidate_answer": "Complete answer",
                        "evaluation_summary": "Perfect, no follow-up needed",
                    }
                )
                
                assert response.status_code == 200
                assert response.json()["follow_up_question"] is None
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_error(self):
        """Test follow-up generation error handling."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("src.api.questions.get_question_generator") as mock_gen:
                mock_generator = MagicMock()
                mock_generator.generate_follow_up = AsyncMock(
                    side_effect=Exception("LLM error")
                )
                mock_gen.return_value = mock_generator
                
                response = await client.post(
                    f"{API_PREFIX}/questions/follow-up",
                    json={
                        "original_question": "Test",
                        "candidate_answer": "Test",
                        "evaluation_summary": "Test",
                    }
                )
                
                assert response.status_code == 500
                assert "Follow-up generation failed" in response.json()["detail"]


class TestMetadataEndpoints:
    """Test metadata/helper endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_stages(self):
        """Test GET /questions/stages."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"{API_PREFIX}/questions/stages")
            
            assert response.status_code == 200
            data = response.json()
            assert "stages" in data
            
            # Check all stages are present
            stage_values = [s["value"] for s in data["stages"]]
            assert "screening" in stage_values
            assert "technical" in stage_values
            assert "behavioral" in stage_values
            assert "system_design" in stage_values
    
    @pytest.mark.asyncio
    async def test_list_difficulties(self):
        """Test GET /questions/difficulties."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"{API_PREFIX}/questions/difficulties")
            
            assert response.status_code == 200
            data = response.json()
            assert "difficulties" in data
            
            # Check all difficulties are present
            diff_values = [d["value"] for d in data["difficulties"]]
            assert "easy" in diff_values
            assert "medium" in diff_values
            assert "hard" in diff_values


class TestRequestValidation:
    """Test request validation and edge cases."""
    
    @pytest.mark.asyncio
    async def test_num_questions_min_validation(self, valid_object_id, sample_jd_doc):
        """Test num_questions minimum validation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"{API_PREFIX}/questions/generate",
                json={
                    "job_description_id": valid_object_id,
                    "stage": "technical",
                    "num_questions": 0,  # Below minimum
                }
            )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_num_questions_max_validation(self, valid_object_id, sample_jd_doc):
        """Test num_questions maximum validation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"{API_PREFIX}/questions/generate",
                json={
                    "job_description_id": valid_object_id,
                    "stage": "technical",
                    "num_questions": 100,  # Above maximum
                }
            )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test missing required fields in requests."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Missing job_description_id
            response = await client.post(
                f"{API_PREFIX}/questions/generate",
                json={
                    "stage": "technical",
                }
            )
            assert response.status_code == 422
            
            # Missing question for evaluation
            response = await client.post(
                f"{API_PREFIX}/answers/evaluate",
                json={
                    "answer": "Some answer",
                }
            )
            assert response.status_code == 422
            
            # Missing answer for evaluation
            response = await client.post(
                f"{API_PREFIX}/answers/evaluate",
                json={
                    "question": "Some question",
                }
            )
            assert response.status_code == 422
