"""
Tests for document processor service.
"""
import pytest
from src.services.document_processor import DocumentProcessor, get_document_processor
from src.models.documents import ParsedResume, ParsedJobDescription


# Sample resume text for testing
SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer
Email: john.doe@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years of experience in Python, JavaScript, and cloud technologies.
Passionate about building scalable systems and machine learning applications.

SKILLS
Python, JavaScript, TypeScript, React, Node.js, Django, FastAPI, PostgreSQL, MongoDB,
AWS, Docker, Kubernetes, Machine Learning, TensorFlow, Git, Agile/Scrum

EXPERIENCE

Senior Software Engineer | TechCorp Inc | San Francisco, CA
January 2021 - Present
- Led development of microservices architecture using Python and FastAPI
- Implemented machine learning pipeline for recommendation system
- Mentored junior developers and conducted code reviews
- Reduced API response time by 40% through optimization

Software Engineer | StartupXYZ | San Jose, CA
June 2018 - December 2020
- Developed full-stack web applications using React and Django
- Built RESTful APIs serving 100K+ daily requests
- Implemented CI/CD pipelines using Jenkins and Docker

EDUCATION

Master of Science in Computer Science
Stanford University | 2018

Bachelor of Science in Computer Engineering
UC Berkeley | 2016

CERTIFICATIONS
- AWS Certified Solutions Architect
- Google Cloud Professional Data Engineer

LANGUAGES
English (Native), Spanish (Intermediate), Mandarin (Basic)
"""


SAMPLE_JD_TEXT = """
Senior Software Engineer - Machine Learning

About the Role:
We are looking for a Senior Software Engineer to join our ML Platform team.

Location: San Francisco, CA
Employment Type: Full-time

Requirements:
- 5+ years of software engineering experience
- Strong proficiency in Python and JavaScript
- Experience with machine learning frameworks (TensorFlow, PyTorch)
- Knowledge of cloud platforms (AWS, GCP, or Azure)
- Experience with Docker and Kubernetes
- Strong understanding of RESTful APIs and microservices

Preferred Qualifications:
- Experience with FastAPI or Django
- Knowledge of PostgreSQL and MongoDB
- Experience with CI/CD pipelines
- Master's degree in Computer Science or related field

Responsibilities:
- Design and implement scalable ML infrastructure
- Build and maintain RESTful APIs for model serving
- Collaborate with data scientists to productionize models
- Write clean, maintainable, and well-tested code
- Mentor junior engineers

Benefits:
- Competitive salary and equity
- Health, dental, and vision insurance
- 401(k) matching
- Flexible work arrangements
"""


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()
    
    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly."""
        assert processor.nlp is not None
        assert len(processor._skill_patterns) > 0
    
    def test_extract_contact_info(self, processor):
        """Test contact information extraction."""
        contact = processor.extract_contact_info(SAMPLE_RESUME_TEXT)
        
        assert contact.email == "john.doe@email.com"
        assert contact.phone == "(555) 123-4567"
        assert "linkedin.com/in/johndoe" in contact.linkedin
        assert "github.com/johndoe" in contact.github
    
    def test_extract_skills(self, processor):
        """Test skill extraction."""
        skills = processor.extract_skills(SAMPLE_RESUME_TEXT)
        
        # Check for expected skills
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "javascript" in skills_lower
        assert "react" in skills_lower
        assert "django" in skills_lower
        assert "fastapi" in skills_lower
        assert "aws" in skills_lower
        assert "docker" in skills_lower
    
    def test_extract_entities(self, processor):
        """Test named entity extraction."""
        entities = processor.extract_entities(SAMPLE_RESUME_TEXT)
        
        # Should find some entities
        assert len(entities) > 0
        
        # Check for PERSON entity (John Doe)
        person_entities = [e for e in entities if e.label == "PERSON"]
        assert len(person_entities) > 0
        
        # Check for ORG entities (companies, universities)
        org_entities = [e for e in entities if e.label == "ORG"]
        assert len(org_entities) > 0
    
    def test_extract_sections(self, processor):
        """Test section extraction."""
        sections = processor.extract_sections(SAMPLE_RESUME_TEXT)
        
        # Should find summary section
        assert "summary" in sections or len(sections) > 0
    
    @pytest.mark.asyncio
    async def test_parse_job_description(self, processor):
        """Test job description parsing."""
        parsed = await processor.parse_job_description(SAMPLE_JD_TEXT)
        
        assert isinstance(parsed, ParsedJobDescription)
        assert len(parsed.required_skills) > 0
        assert parsed.experience_years_min == 5
        
        # Check for required skills
        required_lower = [s.lower() for s in parsed.required_skills]
        assert "python" in required_lower or any("python" in s for s in required_lower)


class TestDocumentProcessorIntegration:
    """Integration tests for document processor."""
    
    def test_get_document_processor_singleton(self):
        """Test that get_document_processor returns singleton."""
        processor1 = get_document_processor()
        processor2 = get_document_processor()
        assert processor1 is processor2


# Run with: pytest tests/test_document_processor.py -v
