"""
Integration test script to verify Week 1 functionality.
Tests document processing and semantic matching with real data.
"""
import asyncio
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
sys.path.insert(0, '.')

from src.services.document_processor import DocumentProcessor
from src.services.semantic_matcher import SemanticMatcher
from src.models.documents import ParsedResume, ParsedJobDescription


# Sample resume text (realistic)
SAMPLE_RESUME = """
SARAH JOHNSON
Senior Software Engineer

Contact Information:
Email: sarah.johnson@techmail.com
Phone: (415) 555-0142
LinkedIn: linkedin.com/in/sarahjohnson-dev
GitHub: github.com/sarahjohnson
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Highly skilled software engineer with 7+ years of experience in full-stack development,
specializing in Python, cloud technologies, and machine learning. Proven track record
of leading teams and delivering scalable solutions at Fortune 500 companies.

TECHNICAL SKILLS
Programming: Python, JavaScript, TypeScript, Java, Go
Frameworks: Django, FastAPI, React, Node.js, Spring Boot
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Cloud & DevOps: AWS, GCP, Docker, Kubernetes, Terraform, CI/CD
ML/AI: TensorFlow, PyTorch, scikit-learn, NLP, Computer Vision

PROFESSIONAL EXPERIENCE

Senior Software Engineer | Google | Mountain View, CA
March 2021 - Present
- Lead a team of 5 engineers developing ML-powered search features
- Architected microservices handling 10M+ daily requests using Python and Go
- Implemented real-time data pipeline reducing latency by 60%
- Mentored junior developers and conducted 50+ technical interviews

Software Engineer II | Meta | Menlo Park, CA  
June 2018 - February 2021
- Built recommendation systems using PyTorch serving 2B+ users
- Developed RESTful APIs with Django handling 5M requests/day
- Optimized PostgreSQL queries reducing response time by 40%
- Led migration of legacy services to Kubernetes

Software Engineer | Startup XYZ | San Francisco, CA
July 2016 - May 2018
- Full-stack development using React and Node.js
- Implemented CI/CD pipelines with Jenkins and Docker
- Built real-time notification system using WebSockets

EDUCATION
Master of Science in Computer Science
Stanford University | 2016
GPA: 3.9/4.0 | Focus: Machine Learning

Bachelor of Science in Computer Engineering  
UC Berkeley | 2014
GPA: 3.7/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect - Professional
- Google Cloud Professional ML Engineer
- Kubernetes Administrator (CKA)

LANGUAGES
English (Native), Mandarin (Fluent), Spanish (Intermediate)
"""


# Sample Job Description (realistic)
SAMPLE_JD = """
Senior Machine Learning Engineer

Company: TechVentures AI
Location: San Francisco, CA (Hybrid)
Employment Type: Full-time

About the Role:
We are seeking a Senior Machine Learning Engineer to join our growing AI team. 
You will be responsible for designing and implementing production ML systems 
that power our core products serving millions of users.

Requirements:
- 5+ years of software engineering experience
- Strong proficiency in Python and machine learning frameworks (TensorFlow, PyTorch)
- Experience building and deploying ML models at scale
- Solid understanding of cloud platforms (AWS, GCP, or Azure)
- Experience with Docker and Kubernetes
- Strong knowledge of SQL and NoSQL databases
- Experience with data pipelines and ETL processes
- Excellent problem-solving and communication skills

Preferred Qualifications:
- Master's degree in Computer Science, Machine Learning, or related field
- Experience with NLP or Computer Vision
- Knowledge of MLOps practices and tools
- Experience mentoring junior engineers
- Publications in ML/AI conferences

Responsibilities:
- Design and implement scalable ML systems
- Build and maintain data pipelines for model training
- Deploy and monitor models in production
- Collaborate with product and research teams
- Mentor junior team members
- Participate in code reviews and architectural discussions

Benefits:
- Competitive salary ($180k - $250k) + equity
- Health, dental, and vision insurance
- 401(k) with 4% match
- Unlimited PTO
- Learning and development budget
"""


# Mismatched resume for comparison
MISMATCHED_RESUME = """
JOHN MARKETING
Marketing Director

Email: john.marketing@email.com
Phone: (212) 555-9999
Location: New York, NY

SUMMARY
Experienced marketing professional with 10 years in digital marketing,
brand strategy, and team leadership.

SKILLS
- Digital Marketing
- SEO/SEM
- Social Media Marketing
- Google Analytics
- Content Strategy
- Brand Management
- Team Leadership
- Budget Management

EXPERIENCE
Marketing Director | BigBrand Corp | 2019-Present
- Led marketing team of 15 people
- Managed $5M annual marketing budget
- Increased brand awareness by 40%

Senior Marketing Manager | AdAgency | 2015-2019
- Developed marketing campaigns
- Managed client relationships
- Drove 30% revenue growth

EDUCATION
MBA, Marketing | NYU Stern | 2015
BA, Communications | Columbia | 2012
"""


async def test_document_processor():
    """Test the document processor with real resume and JD data."""
    print("=" * 60)
    print("TESTING DOCUMENT PROCESSOR")
    print("=" * 60)
    
    processor = DocumentProcessor()
    
    # Test 1: Contact Info Extraction
    print("\n1. Testing Contact Info Extraction...")
    contact = processor.extract_contact_info(SAMPLE_RESUME)
    print(f"   Name: {contact.name}")
    print(f"   Email: {contact.email}")
    print(f"   Phone: {contact.phone}")
    print(f"   LinkedIn: {contact.linkedin}")
    print(f"   GitHub: {contact.github}")
    print(f"   Location: {contact.location}")
    
    # Verify results
    assert contact.email == "sarah.johnson@techmail.com", f"Email mismatch: {contact.email}"
    assert contact.phone == "(415) 555-0142", f"Phone mismatch: {contact.phone}"
    assert "linkedin.com" in (contact.linkedin or ""), "LinkedIn not found"
    assert "github.com" in (contact.github or ""), "GitHub not found"
    print("   [PASS] Contact info extraction PASSED")
    
    # Test 2: Skills Extraction
    print("\n2. Testing Skills Extraction...")
    skills = processor.extract_skills(SAMPLE_RESUME)
    print(f"   Found {len(skills)} skills: {skills[:10]}...")
    
    # Check for expected skills
    expected_skills = ["python", "javascript", "django", "fastapi", "aws", "docker", "kubernetes"]
    found_expected = [s for s in expected_skills if s in skills]
    print(f"   Expected skills found: {found_expected}")
    
    assert len(found_expected) >= 5, f"Missing too many expected skills. Found: {found_expected}"
    print("   âœ“ Skills extraction PASSED")
    
    # Test 3: Entity Extraction
    print("\n3. Testing Named Entity Extraction...")
    entities = processor.extract_entities(SAMPLE_RESUME)
    
    # Group by label
    entity_types = {}
    for ent in entities:
        if ent.label not in entity_types:
            entity_types[ent.label] = []
        entity_types[ent.label].append(ent.text)
    
    print(f"   Found {len(entities)} entities across {len(entity_types)} types")
    for label, texts in list(entity_types.items())[:5]:
        print(f"   - {label}: {texts[:3]}")
    
    assert len(entities) > 10, f"Too few entities found: {len(entities)}"
    assert "PERSON" in entity_types or "ORG" in entity_types, "Missing key entity types"
    print("   âœ“ Entity extraction PASSED")
    
    # Test 4: Job Description Parsing
    print("\n4. Testing Job Description Parsing...")
    jd_parsed = await processor.parse_job_description(SAMPLE_JD)
    
    print(f"   Required Skills ({len(jd_parsed.required_skills)}): {jd_parsed.required_skills[:8]}")
    print(f"   Experience Required: {jd_parsed.experience_years_min}+ years")
    print(f"   Responsibilities: {len(jd_parsed.responsibilities)} items")
    
    assert len(jd_parsed.required_skills) >= 3, "Too few required skills extracted"
    assert jd_parsed.experience_years_min == 5, f"Experience years mismatch: {jd_parsed.experience_years_min}"
    print("   âœ“ JD parsing PASSED")
    
    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSOR: ALL TESTS PASSED âœ“")
    print("=" * 60)
    
    return processor, jd_parsed


async def test_semantic_matcher(processor):
    """Test the semantic matcher with real resume-JD matching."""
    print("\n" + "=" * 60)
    print("TESTING SEMANTIC MATCHER")
    print("=" * 60)
    
    matcher = SemanticMatcher(device="cpu")
    print(f"   Model: {matcher.model_name}")
    print(f"   Device: {matcher.device}")
    
    # Parse resumes
    print("\n1. Parsing test documents...")
    
    # Create parsed resume from sample
    good_resume = ParsedResume(
        summary="Highly skilled software engineer with 7+ years of experience in full-stack development, specializing in Python, cloud technologies, and machine learning.",
        skills=["python", "javascript", "typescript", "java", "go", "django", "fastapi", 
                "react", "node.js", "postgresql", "mongodb", "redis", "aws", "gcp", 
                "docker", "kubernetes", "terraform", "tensorflow", "pytorch", "scikit-learn"],
        raw_text=SAMPLE_RESUME,
    )
    
    bad_resume = ParsedResume(
        summary="Experienced marketing professional with 10 years in digital marketing, brand strategy, and team leadership.",
        skills=["digital marketing", "seo", "social media", "google analytics", 
                "content strategy", "brand management", "team leadership"],
        raw_text=MISMATCHED_RESUME,
    )
    
    jd_parsed = await processor.parse_job_description(SAMPLE_JD)
    
    # Ensure JD has good required skills for matching
    jd_parsed.required_skills = ["python", "tensorflow", "pytorch", "aws", "gcp", 
                                  "docker", "kubernetes", "sql", "machine learning"]
    jd_parsed.experience_years_min = 5
    
    print(f"   Good Resume Skills: {len(good_resume.skills)}")
    print(f"   Bad Resume Skills: {len(bad_resume.skills)}")
    print(f"   JD Required Skills: {jd_parsed.required_skills}")
    
    # Test 2: Semantic Similarity
    print("\n2. Testing Semantic Similarity...")
    
    good_score = matcher.compute_semantic_similarity(good_resume, jd_parsed)
    bad_score = matcher.compute_semantic_similarity(bad_resume, jd_parsed)
    
    print(f"   Good Resume Semantic Score: {good_score:.1f}/100")
    print(f"   Bad Resume Semantic Score: {bad_score:.1f}/100")
    print(f"   Difference: {good_score - bad_score:.1f} points")
    
    assert good_score > bad_score, f"Good resume should score higher! Good: {good_score}, Bad: {bad_score}"
    print("   âœ“ Semantic similarity correctly differentiates candidates")
    
    # Test 3: Skill Matching
    print("\n3. Testing Skill Matching...")
    
    good_skill_score, good_matched, good_missing = matcher.compute_skill_match(good_resume, jd_parsed)
    bad_skill_score, bad_matched, bad_missing = matcher.compute_skill_match(bad_resume, jd_parsed)
    
    print(f"   Good Resume Skill Score: {good_skill_score:.1f}/100")
    print(f"   - Matched: {good_matched}")
    print(f"   - Missing: {good_missing}")
    print(f"   Bad Resume Skill Score: {bad_skill_score:.1f}/100")
    print(f"   - Matched: {bad_matched}")
    print(f"   - Missing: {bad_missing}")
    
    assert good_skill_score > bad_skill_score, "Good resume should have better skill match"
    assert len(good_matched) > len(bad_matched), "Good resume should have more skill matches"
    print("   âœ“ Skill matching correctly identifies relevant skills")
    
    # Test 4: Full Match Pipeline
    print("\n4. Testing Full Match Pipeline...")
    
    good_result = await matcher.match(good_resume, jd_parsed, "resume-good", "jd-001")
    bad_result = await matcher.match(bad_resume, jd_parsed, "resume-bad", "jd-001")
    
    print(f"\n   GOOD CANDIDATE MATCH RESULT:")
    print(f"   - Overall Score: {good_result.overall_score:.1f}/100")
    print(f"   - Skill Match: {good_result.skill_match_score:.1f}/100")
    print(f"   - Experience Match: {good_result.experience_match_score:.1f}/100")
    print(f"   - Semantic Similarity: {good_result.semantic_similarity_score:.1f}/100")
    print(f"   - Matched Skills: {good_result.matched_skills}")
    print(f"   - Missing Skills: {good_result.missing_skills}")
    print(f"   - Recommendations: {good_result.recommendations}")
    
    print(f"\n   BAD CANDIDATE MATCH RESULT:")
    print(f"   - Overall Score: {bad_result.overall_score:.1f}/100")
    print(f"   - Skill Match: {bad_result.skill_match_score:.1f}/100")
    print(f"   - Experience Match: {bad_result.experience_match_score:.1f}/100")
    print(f"   - Semantic Similarity: {bad_result.semantic_similarity_score:.1f}/100")
    print(f"   - Matched Skills: {bad_result.matched_skills}")
    print(f"   - Missing Skills: {bad_result.missing_skills}")
    
    # Verify overall matching logic
    assert good_result.overall_score > bad_result.overall_score, \
        f"Good candidate should score higher! Good: {good_result.overall_score}, Bad: {bad_result.overall_score}"
    
    score_diff = good_result.overall_score - bad_result.overall_score
    print(f"\n   Score Difference: {score_diff:.1f} points")
    assert score_diff > 20, f"Score difference should be substantial: {score_diff}"
    
    print("   âœ“ Full matching pipeline correctly ranks candidates")
    
    # Test 5: Embedding Generation
    print("\n5. Testing Embedding Generation...")
    
    embedding = await matcher.compute_embedding("Python developer with machine learning experience")
    print(f"   Embedding dimensions: {len(embedding)}")
    print(f"   Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
    
    assert len(embedding) == 1024, f"BGE-large should produce 1024-dim embeddings, got {len(embedding)}"
    print("   âœ“ Embedding generation produces correct dimensions")
    
    # Test 6: Similar Resume Search
    print("\n6. Testing Similar Resume Search...")
    
    jd_emb = await matcher.compute_embedding(
        "Senior ML Engineer with Python, TensorFlow, and cloud experience"
    )
    
    resume_embeddings = [
        ("resume-ml-expert", await matcher.compute_embedding(
            "Machine learning engineer with 8 years Python experience, TensorFlow, AWS"
        )),
        ("resume-marketing", await matcher.compute_embedding(
            "Marketing manager with SEO and social media expertise"
        )),
        ("resume-frontend", await matcher.compute_embedding(
            "Frontend developer specializing in React and TypeScript"
        )),
        ("resume-ml-junior", await matcher.compute_embedding(
            "Junior data scientist with Python and scikit-learn experience"
        )),
    ]
    
    results = await matcher.find_similar_resumes(jd_emb, resume_embeddings, top_k=4)
    
    print("   Ranking results:")
    for i, (resume_id, score) in enumerate(results, 1):
        print(f"   {i}. {resume_id}: {score:.3f}")
    
    # ML expert should rank first
    assert results[0][0] == "resume-ml-expert", f"ML expert should rank first, got {results[0][0]}"
    # Marketing should rank last or near last
    marketing_rank = [r[0] for r in results].index("resume-marketing") + 1
    assert marketing_rank >= 3, f"Marketing resume should rank low, got rank {marketing_rank}"
    
    print("   âœ“ Similar resume search correctly ranks by relevance")
    
    print("\n" + "=" * 60)
    print("SEMANTIC MATCHER: ALL TESTS PASSED âœ“")
    print("=" * 60)


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("AI INTERVIEWER SYSTEM - WEEK 1 INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Test document processor
        processor, jd_parsed = await test_document_processor()
        
        # Test semantic matcher
        await test_semantic_matcher(processor)
        
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED! âœ“")
        print("=" * 60)
        print("\nWeek 1 Implementation Status:")
        print("  âœ“ Document processing works correctly")
        print("  âœ“ Contact/skill/entity extraction functional")
        print("  âœ“ Semantic matching differentiates candidates")
        print("  âœ“ Skill matching identifies relevant skills")
        print("  âœ“ BGE-large embeddings (1024-dim) working")
        print("  âœ“ Resume ranking by relevance works")
        print("\nReady for Week 2: LLM integration & question generation")
        
    except AssertionError as e:
        print(f"\nâŒ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nâŒ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

