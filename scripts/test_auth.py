#!/usr/bin/env python
"""
Test script for JWT authentication implementation.

Tests:
1. Token creation and decoding
2. Role-based access control
3. Permission-based access control
4. Session-based access for candidates
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import timedelta

from src.core.auth import create_token, decode_token, get_auth_config
from src.core.permissions import (
    get_permissions_for_role,
    has_permission,
    Permissions,
)
from src.models.auth import UserRole, TokenPayload, AuthenticatedUser


def test_token_creation():
    """Test JWT token creation."""
    print("\n=== Testing Token Creation ===")
    
    # Create admin token
    admin_token = create_token(
        subject="admin-001",
        role=UserRole.ADMIN,
    )
    print(f"Admin token created: {admin_token[:50]}...")
    
    # Create candidate token with session
    candidate_token = create_token(
        subject="candidate-001",
        role=UserRole.CANDIDATE,
        session_id="interview-session-123",
    )
    print(f"Candidate token created: {candidate_token[:50]}...")
    
    return admin_token, candidate_token


def test_token_decoding(admin_token: str, candidate_token: str):
    """Test JWT token decoding."""
    print("\n=== Testing Token Decoding ===")
    
    # Decode admin token
    admin_payload = decode_token(admin_token)
    print(f"Admin payload: sub={admin_payload.sub}, role={admin_payload.role}")
    assert admin_payload.sub == "admin-001"
    assert admin_payload.role == UserRole.ADMIN
    
    # Decode candidate token
    candidate_payload = decode_token(candidate_token)
    print(f"Candidate payload: sub={candidate_payload.sub}, role={candidate_payload.role}, session_id={candidate_payload.session_id}")
    assert candidate_payload.sub == "candidate-001"
    assert candidate_payload.role == UserRole.CANDIDATE
    assert candidate_payload.session_id == "interview-session-123"
    
    print("Token decoding: PASSED")


def test_role_permissions():
    """Test role-based permissions."""
    print("\n=== Testing Role Permissions ===")
    
    # Test admin permissions
    admin_perms = get_permissions_for_role(UserRole.ADMIN)
    print(f"Admin permissions: {admin_perms}")
    assert Permissions.ALL in admin_perms
    
    # Test hiring manager permissions
    hm_perms = get_permissions_for_role(UserRole.HIRING_MANAGER)
    print(f"Hiring Manager permissions: {len(hm_perms)} permissions")
    assert Permissions.CREATE_INTERVIEW in hm_perms
    assert Permissions.VIEW_REPORTS in hm_perms
    
    # Test interviewer permissions
    interviewer_perms = get_permissions_for_role(UserRole.INTERVIEWER)
    print(f"Interviewer permissions: {len(interviewer_perms)} permissions")
    assert Permissions.CONDUCT_INTERVIEW in interviewer_perms
    assert Permissions.CREATE_INTERVIEW not in interviewer_perms
    
    # Test candidate permissions
    candidate_perms = get_permissions_for_role(UserRole.CANDIDATE)
    print(f"Candidate permissions: {candidate_perms}")
    assert Permissions.PARTICIPATE_SESSION in candidate_perms
    assert len(candidate_perms) == 2  # PARTICIPATE_SESSION and USE_VOICE
    
    print("Role permissions: PASSED")


def test_permission_checks():
    """Test permission checking functions."""
    print("\n=== Testing Permission Checks ===")
    
    # Admin should have all permissions
    assert has_permission(UserRole.ADMIN, Permissions.CREATE_INTERVIEW) == True
    assert has_permission(UserRole.ADMIN, Permissions.DELETE_SESSION) == True
    assert has_permission(UserRole.ADMIN, "any_random_permission") == True  # ALL covers everything
    print("Admin permission checks: PASSED")
    
    # Hiring manager specific permissions
    assert has_permission(UserRole.HIRING_MANAGER, Permissions.CREATE_INTERVIEW) == True
    assert has_permission(UserRole.HIRING_MANAGER, Permissions.VIEW_REPORTS) == True
    print("Hiring Manager permission checks: PASSED")
    
    # Interviewer limited permissions
    assert has_permission(UserRole.INTERVIEWER, Permissions.CONDUCT_INTERVIEW) == True
    assert has_permission(UserRole.INTERVIEWER, Permissions.CREATE_INTERVIEW) == False
    assert has_permission(UserRole.INTERVIEWER, Permissions.DELETE_SESSION) == False
    print("Interviewer permission checks: PASSED")
    
    # Candidate minimal permissions
    assert has_permission(UserRole.CANDIDATE, Permissions.PARTICIPATE_SESSION) == True
    assert has_permission(UserRole.CANDIDATE, Permissions.VIEW_REPORTS) == False
    print("Candidate permission checks: PASSED")


def test_authenticated_user():
    """Test AuthenticatedUser model methods."""
    print("\n=== Testing AuthenticatedUser Model ===")
    
    # Create admin user
    admin_user = AuthenticatedUser(
        user_id="admin-001",
        role=UserRole.ADMIN,
        permissions=[Permissions.ALL],
        token_exp=9999999999,
    )
    
    assert admin_user.has_permission(Permissions.CREATE_INTERVIEW) == True
    assert admin_user.can_access_session("any-session-id") == True
    print("Admin user checks: PASSED")
    
    # Create candidate user with session
    candidate_user = AuthenticatedUser(
        user_id="candidate-001",
        role=UserRole.CANDIDATE,
        permissions=[Permissions.PARTICIPATE_SESSION, Permissions.USE_VOICE],
        session_id="interview-session-123",
        token_exp=9999999999,
    )
    
    assert candidate_user.has_permission(Permissions.PARTICIPATE_SESSION) == True
    assert candidate_user.has_permission(Permissions.VIEW_REPORTS) == False
    assert candidate_user.can_access_session("interview-session-123") == True
    assert candidate_user.can_access_session("other-session") == False
    print("Candidate user checks: PASSED")
    
    # Create interviewer with session permission
    interviewer_user = AuthenticatedUser(
        user_id="interviewer-001",
        role=UserRole.INTERVIEWER,
        permissions=[
            Permissions.CONDUCT_INTERVIEW,
            Permissions.VIEW_SESSION,
            "session:interview-session-456",  # Explicit session access
        ],
        token_exp=9999999999,
    )
    
    assert interviewer_user.can_access_session("interview-session-456") == True
    assert interviewer_user.can_access_session("other-session") == False
    print("Interviewer user checks: PASSED")


def test_config():
    """Test auth configuration."""
    print("\n=== Testing Auth Configuration ===")
    
    config = get_auth_config()
    print(f"Algorithm: {config.algorithm}")
    print(f"Verify exp: {config.verify_exp}")
    print(f"Leeway: {config.leeway} seconds")
    print(f"Public paths: {config.public_paths}")
    
    assert config.algorithm == "HS256"
    assert config.verify_exp == True
    print("Auth configuration: PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("JWT Authentication Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        admin_token, candidate_token = test_token_creation()
        test_token_decoding(admin_token, candidate_token)
        test_role_permissions()
        test_permission_checks()
        test_authenticated_user()
        test_config()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
