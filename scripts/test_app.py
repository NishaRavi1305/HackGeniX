"""
Test that FastAPI app can be imported and configured correctly.
"""
import sys
sys.path.insert(0, '.')

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("Testing FastAPI Application Creation...")
    
    # Test 1: Import main module
    print("\n1. Testing module imports...")
    try:
        from src.core.config import get_settings, load_model_config
        settings = get_settings()
        print(f"   Settings loaded: {settings.app_name}")
        print(f"   Environment: {settings.app_env}")
        print(f"   MongoDB URI: {settings.mongodb_uri}")
        print(f"   [PASS] Config module works")
    except Exception as e:
        print(f"   [FAIL] Config error: {e}")
        return False
    
    # Test 2: Load model config
    print("\n2. Testing model config...")
    try:
        config = load_model_config()
        print(f"   LLM: {config['providers']['llm']['model']}")
        print(f"   Embeddings: {config['providers']['embeddings']['model']}")
        print(f"   STT: {config['providers']['stt']['model']}")
        print(f"   TTS: {config['providers']['tts']['provider']}")
        print(f"   [PASS] Model config loads correctly")
    except Exception as e:
        print(f"   [FAIL] Model config error: {e}")
        return False
    
    # Test 3: Import API modules
    print("\n3. Testing API module imports...")
    try:
        from src.api import health, documents, interviews
        print(f"   Health router: {len(health.router.routes)} routes")
        print(f"   Documents router: {len(documents.router.routes)} routes")
        print(f"   Interviews router: {len(interviews.router.routes)} routes")
        print(f"   [PASS] API modules import correctly")
    except Exception as e:
        print(f"   [FAIL] API import error: {e}")
        return False
    
    # Test 4: Create FastAPI app (without starting server)
    print("\n4. Testing FastAPI app creation...")
    try:
        from src.main import create_app
        app = create_app()
        print(f"   App title: {app.title}")
        print(f"   App version: {app.version}")
        print(f"   Total routes: {len(app.routes)}")
        
        # List routes
        print("   Routes:")
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ','.join(route.methods - {'HEAD', 'OPTIONS'}) if route.methods else ''
                if methods:
                    print(f"     {methods:8} {route.path}")
        
        print(f"   [PASS] FastAPI app creates successfully")
    except Exception as e:
        print(f"   [FAIL] App creation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test Pydantic models
    print("\n5. Testing Pydantic models...")
    try:
        from src.models.documents import (
            ResumeDocument, JobDescriptionDocument, 
            MatchResult, ParsedResume, ParsedJobDescription
        )
        
        # Create sample instances
        resume = ParsedResume(
            skills=["python", "javascript"],
            raw_text="Sample resume"
        )
        print(f"   ParsedResume: {len(resume.skills)} skills")
        
        jd = ParsedJobDescription(
            required_skills=["python"],
            raw_text="Sample JD"
        )
        print(f"   ParsedJobDescription: {len(jd.required_skills)} required skills")
        
        match = MatchResult(
            resume_id="r1",
            job_description_id="jd1",
            overall_score=75.5,
            skill_match_score=80,
            experience_match_score=70,
            semantic_similarity_score=76
        )
        print(f"   MatchResult: {match.overall_score} overall score")
        
        print(f"   [PASS] Pydantic models work correctly")
    except Exception as e:
        print(f"   [FAIL] Model error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("FASTAPI APPLICATION: ALL CHECKS PASSED!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_app_creation()
    sys.exit(0 if success else 1)
