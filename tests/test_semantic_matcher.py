"""
Tests for semantic matcher service.
"""
import pytest
import numpy as np
from src.services.semantic_matcher import SemanticMatcher, get_semantic_matcher
from src.models.documents import ParsedResume, ParsedJobDescription, ContactInfo, Experience


# Sample data for testing
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


def create_mismatched_resume() -> ParsedResume:
    """Create a resume that doesn't match the sample JD well."""
    return ParsedResume(
        contact=ContactInfo(name="Jane Smith"),
        summary="Marketing professional with expertise in digital campaigns.",
        skills=["marketing", "seo", "google analytics", "social media", "content writing"],
        experience=[
            Experience(
                company="MarketingCo",
                title="Marketing Manager",
                description="Led digital marketing campaigns",
            ),
        ],
        raw_text="Jane Smith - Marketing professional...",
    )


class TestSemanticMatcher:
    """Tests for SemanticMatcher class."""
    
    @pytest.fixture
    def matcher(self):
        """Create a semantic matcher instance with CPU for testing."""
        # Use CPU for testing to avoid CUDA issues
        return SemanticMatcher(device="cpu")
    
    def test_matcher_initialization(self, matcher):
        """Test that matcher initializes correctly."""
        assert matcher.model is not None
        assert matcher.device == "cpu"
    
    def test_encode_single_text(self, matcher):
        """Test encoding a single text."""
        embeddings = matcher.encode(["This is a test sentence."])
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 100  # Embedding dimension should be substantial
    
    def test_encode_multiple_texts(self, matcher):
        """Test encoding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = matcher.encode(texts)
        
        assert embeddings.shape[0] == 3
    
    def test_cosine_similarity_identical(self, matcher):
        """Test cosine similarity of identical vectors."""
        embedding = matcher.encode(["Test sentence"])[0]
        similarity = matcher.cosine_similarity(embedding, embedding)
        
        # Identical vectors should have similarity close to 1
        assert similarity > 0.99
    
    def test_cosine_similarity_different(self, matcher):
        """Test cosine similarity of different vectors."""
        emb1 = matcher.encode(["Python programming language"])[0]
        emb2 = matcher.encode(["Cooking delicious pasta"])[0]
        
        similarity = matcher.cosine_similarity(emb1, emb2)
        
        # Very different texts should have lower similarity (but not necessarily < 0.5)
        # BGE models can have higher baseline similarity
        assert similarity < 0.7
    
    def test_compute_semantic_similarity_good_match(self, matcher):
        """Test semantic similarity for a good resume-JD match."""
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        score = matcher.compute_semantic_similarity(resume, jd)
        
        # Good match should have reasonable similarity
        assert score > 30  # At least 30% similar
        assert score <= 100
    
    def test_compute_semantic_similarity_poor_match(self, matcher):
        """Test semantic similarity for a poor resume-JD match."""
        resume = create_mismatched_resume()
        jd = create_sample_jd()
        
        good_resume = create_sample_resume()
        
        poor_score = matcher.compute_semantic_similarity(resume, jd)
        good_score = matcher.compute_semantic_similarity(good_resume, jd)
        
        # Poor match should score lower than good match
        assert poor_score < good_score
    
    def test_compute_skill_match_good_overlap(self, matcher):
        """Test skill matching with good overlap."""
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        score, matched, missing = matcher.compute_skill_match(resume, jd)
        
        # Should have some matched skills
        assert len(matched) > 0
        assert score > 50  # More than half of skills should match
        
        # Python and Django should match
        matched_lower = [s.lower() for s in matched]
        assert "python" in matched_lower
        assert "django" in matched_lower
    
    def test_compute_skill_match_poor_overlap(self, matcher):
        """Test skill matching with poor overlap."""
        resume = create_mismatched_resume()
        jd = create_sample_jd()
        
        score, matched, missing = matcher.compute_skill_match(resume, jd)
        
        # Should have few or no matched skills
        assert len(missing) > len(matched)
        assert score < 50
    
    def test_compute_experience_match(self, matcher):
        """Test experience matching."""
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        score = matcher.compute_experience_match(resume, jd)
        
        # Resume has 2 experience entries, JD requires 5 years
        # Should give a reasonable score
        assert score >= 0
        assert score <= 100
    
    def test_generate_recommendations(self, matcher):
        """Test recommendation generation."""
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        recommendations = matcher.generate_recommendations(
            resume, jd, ["kubernetes"], 75.0
        )
        
        assert len(recommendations) > 0
        # Should mention missing skills
        assert any("kubernetes" in r.lower() for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_match_full_pipeline(self, matcher):
        """Test full matching pipeline."""
        resume = create_sample_resume()
        jd = create_sample_jd()
        
        result = await matcher.match(resume, jd, "resume-123", "jd-456")
        
        assert result.resume_id == "resume-123"
        assert result.job_description_id == "jd-456"
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        assert result.skill_match_score >= 0
        assert result.semantic_similarity_score >= 0
        assert isinstance(result.matched_skills, list)
        assert isinstance(result.missing_skills, list)
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_compute_embedding(self, matcher):
        """Test single embedding computation."""
        embedding = await matcher.compute_embedding("Python developer with ML experience")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 100
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_find_similar_resumes(self, matcher):
        """Test finding similar resumes."""
        # Create JD embedding
        jd_embedding = await matcher.compute_embedding(
            "Looking for Python developer with Django experience"
        )
        
        # Create some resume embeddings
        resume_embeddings = []
        resume_embeddings.append((
            "resume-1",
            await matcher.compute_embedding("Python Django developer with 5 years experience")
        ))
        resume_embeddings.append((
            "resume-2",
            await matcher.compute_embedding("Marketing manager with SEO skills")
        ))
        resume_embeddings.append((
            "resume-3",
            await matcher.compute_embedding("Python Flask developer looking for new opportunities")
        ))
        
        results = await matcher.find_similar_resumes(jd_embedding, resume_embeddings, top_k=2)
        
        assert len(results) == 2
        # Python resumes should rank higher than marketing resume
        resume_ids = [r[0] for r in results]
        assert "resume-2" not in resume_ids  # Marketing resume should not be in top 2


class TestSemanticMatcherIntegration:
    """Integration tests for semantic matcher."""
    
    def test_get_semantic_matcher_singleton(self):
        """Test that get_semantic_matcher returns singleton."""
        # Reset global state first
        import src.services.semantic_matcher as sm
        sm._matcher = None
        
        matcher1 = get_semantic_matcher()
        matcher2 = get_semantic_matcher()
        assert matcher1 is matcher2
        
        # Cleanup
        sm._matcher = None


# Run with: pytest tests/test_semantic_matcher.py -v
