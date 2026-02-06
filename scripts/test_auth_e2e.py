#!/usr/bin/env python
"""
End-to-end auth test with self-generated JWT key.

Tests the full auth flow:
1. Generate a secure key
2. Create tokens for different roles
3. Test token validation
4. Simulate API endpoint access with different roles
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set a test JWT secret before importing auth modules
TEST_JWT_SECRET = "TestSecretKey_HYcVL4tfsKJUnYcK8H_tVRv5DNTZQGXN2PMb8C3q8bg"
os.environ["JWT_SECRET_KEY"] = TEST_JWT_SECRET
os.environ["AUTH_ENABLED"] = "true"

import secrets
from datetime import timedelta
from unittest.mock import MagicMock, AsyncMock

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.core.auth import create_token, decode_token, get_current_user, require_role, require_permission
from src.core.permissions import Permissions, get_permissions_for_role
from src.models.auth import UserRole, AuthenticatedUser


def generate_test_key():
    """Generate a cryptographically secure key."""
    key = secrets.token_urlsafe(32)
    print(f"Generated Test Key: {key[:20]}...{key[-10:]}")
    print(f"Key Length: {len(key)} characters")
    return key


def test_token_lifecycle():
    """Test complete token lifecycle with different roles."""
    print("\n" + "="*60)
    print("TOKEN LIFECYCLE TEST")
    print("="*60)
    
    roles_to_test = [
        (UserRole.ADMIN, "admin-user-001", None),
        (UserRole.HIRING_MANAGER, "hm-user-002", None),
        (UserRole.INTERVIEWER, "interviewer-003", None),
        (UserRole.CANDIDATE, "candidate-004", "session-abc-123"),
    ]
    
    tokens = {}
    
    for role, user_id, session_id in roles_to_test:
        print(f"\n--- Testing {role.value.upper()} ---")
        
        # Create token
        token = create_token(
            subject=user_id,
            role=role,
            session_id=session_id,
            expires_delta=timedelta(hours=1),
        )
        tokens[role] = token
        print(f"Token created: {token[:50]}...")
        
        # Decode and validate
        payload = decode_token(token)
        print(f"Decoded: sub={payload.sub}, role={payload.role.value}, session_id={payload.session_id}")
        
        # Verify claims
        assert payload.sub == user_id, f"Subject mismatch: {payload.sub} != {user_id}"
        assert payload.role == role, f"Role mismatch: {payload.role} != {role}"
        if session_id:
            assert payload.session_id == session_id, f"Session mismatch"
        
        # Check permissions
        perms = get_permissions_for_role(role)
        print(f"Permissions: {len(perms)} granted")
        
    print("\n[PASS] Token lifecycle test passed for all roles")
    return tokens


async def test_auth_dependencies(tokens: dict):
    """Test FastAPI auth dependencies."""
    print("\n" + "="*60)
    print("AUTH DEPENDENCIES TEST")
    print("="*60)
    
    # Test get_current_user with valid token
    print("\n--- Testing get_current_user ---")
    
    admin_creds = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=tokens[UserRole.ADMIN]
    )
    
    user = await get_current_user(admin_creds)
    print(f"Authenticated user: {user.user_id}, role={user.role.value}")
    assert user.role == UserRole.ADMIN
    assert user.has_permission(Permissions.ALL)
    print("[PASS] get_current_user works")
    
    # Test require_role
    print("\n--- Testing require_role ---")
    
    admin_only = require_role(UserRole.ADMIN)
    user = await admin_only(await get_current_user(admin_creds))
    print(f"Admin-only check passed for: {user.user_id}")
    
    # Test candidate trying admin endpoint
    candidate_creds = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=tokens[UserRole.CANDIDATE]
    )
    
    try:
        candidate_user = await get_current_user(candidate_creds)
        await admin_only(candidate_user)
        print("[FAIL] Candidate should not access admin endpoint")
        assert False
    except HTTPException as e:
        print(f"[PASS] Candidate correctly blocked: {e.detail}")
    
    # Test require_permission
    print("\n--- Testing require_permission ---")
    
    view_reports = require_permission(Permissions.VIEW_REPORTS)
    
    # Hiring manager should have VIEW_REPORTS
    hm_creds = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=tokens[UserRole.HIRING_MANAGER]
    )
    hm_user = await get_current_user(hm_creds)
    user = await view_reports(hm_user)
    print(f"[PASS] Hiring manager can view reports")
    
    # Candidate should NOT have VIEW_REPORTS
    try:
        candidate_user = await get_current_user(candidate_creds)
        await view_reports(candidate_user)
        print("[FAIL] Candidate should not view reports")
        assert False
    except HTTPException as e:
        print(f"[PASS] Candidate correctly blocked from reports: {e.detail}")


def test_session_access():
    """Test session-based access for candidates."""
    print("\n" + "="*60)
    print("SESSION ACCESS TEST")
    print("="*60)
    
    # Create candidate with specific session
    session_id = "interview-session-xyz-789"
    token = create_token(
        subject="candidate-test",
        role=UserRole.CANDIDATE,
        session_id=session_id,
    )
    
    payload = decode_token(token)
    
    # Create AuthenticatedUser
    user = AuthenticatedUser(
        user_id=payload.sub,
        role=payload.role,
        permissions=get_permissions_for_role(payload.role),
        session_id=payload.session_id,
        token_exp=payload.exp,
    )
    
    # Test session access
    print(f"\nCandidate session: {user.session_id}")
    
    # Can access own session
    assert user.can_access_session(session_id) == True
    print(f"[PASS] Can access own session: {session_id}")
    
    # Cannot access other sessions
    assert user.can_access_session("other-session") == False
    print(f"[PASS] Cannot access other session: other-session")
    
    # Admin can access any session
    admin_user = AuthenticatedUser(
        user_id="admin",
        role=UserRole.ADMIN,
        permissions=[Permissions.ALL],
        token_exp=9999999999,
    )
    assert admin_user.can_access_session(session_id) == True
    assert admin_user.can_access_session("any-random-session") == True
    print(f"[PASS] Admin can access any session")


def test_invalid_tokens():
    """Test handling of invalid tokens."""
    print("\n" + "="*60)
    print("INVALID TOKEN TEST")
    print("="*60)
    
    # Test expired token (we can't easily create one, but we test malformed)
    
    # Test malformed token
    print("\n--- Testing malformed token ---")
    try:
        decode_token("not.a.valid.token")
        print("[FAIL] Should reject malformed token")
        assert False
    except HTTPException as e:
        print(f"[PASS] Malformed token rejected: {e.detail}")
    
    # Test token with wrong secret
    print("\n--- Testing token with wrong secret ---")
    import jwt
    wrong_token = jwt.encode(
        {"sub": "hacker", "exp": 9999999999, "role": "admin"},
        "wrong-secret-key-completely-different",
        algorithm="HS256"
    )
    try:
        decode_token(wrong_token)
        print("[FAIL] Should reject token with wrong secret")
        assert False
    except HTTPException as e:
        print(f"[PASS] Wrong secret token rejected: {e.detail}")
    
    # Test empty token
    print("\n--- Testing empty token ---")
    try:
        decode_token("")
        print("[FAIL] Should reject empty token")
        assert False
    except HTTPException as e:
        print(f"[PASS] Empty token rejected: {e.detail}")


def test_permission_hierarchy():
    """Test permission hierarchy and checks."""
    print("\n" + "="*60)
    print("PERMISSION HIERARCHY TEST")
    print("="*60)
    
    # Test each role's permissions
    test_cases = [
        (UserRole.ADMIN, Permissions.ALL, True),
        (UserRole.ADMIN, Permissions.DELETE_SESSION, True),  # ALL covers everything
        (UserRole.HIRING_MANAGER, Permissions.CREATE_INTERVIEW, True),
        (UserRole.HIRING_MANAGER, Permissions.VIEW_REPORTS, True),
        (UserRole.INTERVIEWER, Permissions.CONDUCT_INTERVIEW, True),
        (UserRole.INTERVIEWER, Permissions.CREATE_INTERVIEW, False),
        (UserRole.INTERVIEWER, Permissions.DELETE_SESSION, False),
        (UserRole.CANDIDATE, Permissions.PARTICIPATE_SESSION, True),
        (UserRole.CANDIDATE, Permissions.USE_VOICE, True),
        (UserRole.CANDIDATE, Permissions.VIEW_REPORTS, False),
        (UserRole.CANDIDATE, Permissions.CREATE_INTERVIEW, False),
    ]
    
    from src.core.permissions import has_permission
    
    for role, permission, expected in test_cases:
        result = has_permission(role, permission)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {role.value}.has({permission}) = {result} (expected: {expected})")
        assert result == expected, f"Permission check failed for {role.value}.{permission}"
    
    print("\n[PASS] All permission hierarchy checks passed")


async def main():
    """Run all tests."""
    print("="*60)
    print("JWT AUTHENTICATION END-TO-END TEST")
    print(f"Using test secret key: {TEST_JWT_SECRET[:20]}...")
    print("="*60)
    
    try:
        # Run tests
        tokens = test_token_lifecycle()
        await test_auth_dependencies(tokens)
        test_session_access()
        test_invalid_tokens()
        test_permission_hierarchy()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nJWT Authentication is working correctly with:")
        print(f"  - Secret Key: {TEST_JWT_SECRET[:20]}...")
        print(f"  - Algorithm: HS256")
        print(f"  - Roles: Admin, Hiring Manager, Interviewer, Candidate")
        print(f"  - Session-based access for candidates")
        print(f"  - Permission-based endpoint protection")
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
