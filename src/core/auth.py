"""
Core authentication module for the AI Interviewer System.

Provides plug-and-play JWT authentication that integrates with
the client's existing auth system.

Usage:
    1. Client's backend issues JWTs with their secret
    2. Configure JWT_SECRET_KEY in our .env to match
    3. All protected endpoints automatically validate tokens

The client just needs to:
    - Issue JWTs with: sub, exp, role claims
    - Share the JWT secret with us
    - Include "Authorization: Bearer <token>" in requests
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.config import get_settings
from src.models.auth import (
    TokenPayload,
    AuthenticatedUser,
    AuthConfig,
    UserRole,
)

logger = logging.getLogger(__name__)

# Security scheme for Swagger UI
security_scheme = HTTPBearer(
    scheme_name="JWT",
    description="JWT token from client's auth system. Format: Bearer <token>",
    auto_error=False,  # Don't auto-raise, we handle it for better error messages
)


def get_auth_config() -> AuthConfig:
    """
    Load auth configuration from settings.
    
    This makes auth plug-and-play - just set environment variables.
    """
    settings = get_settings()
    
    return AuthConfig(
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        verify_exp=True,
        verify_iss=False,  # Set to True if client provides issuer
        verify_aud=False,  # Set to True if client provides audience
        expected_issuer=None,
        expected_audience="ai-interviewer",
        leeway=30,
    )


def decode_token(token: str, config: Optional[AuthConfig] = None) -> TokenPayload:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token string
        config: Auth configuration (uses default if not provided)
        
    Returns:
        TokenPayload with decoded claims
        
    Raises:
        HTTPException: If token is invalid, expired, or malformed
    """
    if config is None:
        config = get_auth_config()
    
    try:
        # Build decode options
        options = {
            "verify_exp": config.verify_exp,
            "verify_iss": config.verify_iss,
            "verify_aud": config.verify_aud,
            "require": ["sub", "exp"],
        }
        
        # Build verification kwargs
        decode_kwargs = {
            "algorithms": [config.algorithm],
            "options": options,
            "leeway": timedelta(seconds=config.leeway),
        }
        
        if config.verify_iss and config.expected_issuer:
            decode_kwargs["issuer"] = config.expected_issuer
            
        if config.verify_aud and config.expected_audience:
            decode_kwargs["audience"] = config.expected_audience
        
        # Decode token
        payload = jwt.decode(
            token,
            config.secret_key,
            **decode_kwargs
        )
        
        # Parse role from string if present
        if "role" in payload and isinstance(payload["role"], str):
            payload["role"] = UserRole.from_string(payload["role"])
        
        return TokenPayload(**payload)
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        logger.warning("Invalid token issuer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        logger.warning("Invalid token audience")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.DecodeError as e:
        logger.warning(f"Token decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_token(
    subject: str,
    role: UserRole = UserRole.CANDIDATE,
    session_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[dict] = None,
) -> str:
    """
    Create a JWT token.
    
    This is provided for convenience/testing. In production,
    the client's backend should issue tokens.
    
    Args:
        subject: User ID or session identifier
        role: User role
        session_id: Interview session ID (for candidates)
        expires_delta: Token lifetime (default: from settings)
        additional_claims: Extra claims to include
        
    Returns:
        Encoded JWT token string
    """
    settings = get_settings()
    config = get_auth_config()
    
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    # Use time.time() for consistent timestamps across all systems
    now_ts = int(time.time())
    expire_ts = now_ts + int(expires_delta.total_seconds())
    
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": now_ts,
        "exp": expire_ts,
        "iss": "ai-interviewer",
        "aud": config.expected_audience,
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    if additional_claims:
        payload.update(additional_claims)
    
    return jwt.encode(payload, config.secret_key, algorithm=config.algorithm)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.
    
    Usage in route handlers:
        @app.get("/protected")
        async def protected_route(user: AuthenticatedUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    
    When auth_enabled=False in settings, returns a mock admin user for development.
    """
    settings = get_settings()
    
    # If auth is disabled, return a mock admin user for development
    if not settings.auth_enabled:
        from src.core.permissions import Permissions
        return AuthenticatedUser(
            user_id="dev-user",
            role=UserRole.ADMIN,
            permissions=[Permissions.ALL],
            session_id=None,
            name="Development User",
            email="dev@localhost",
            token_exp=9999999999,
            token_iss="dev-mode",
        )
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_payload = decode_token(credentials.credentials)
    
    # Import here to avoid circular imports
    from src.core.permissions import get_permissions_for_role
    
    # Get effective permissions (explicit permissions override role-based)
    if token_payload.permissions:
        effective_permissions = token_payload.permissions
    else:
        effective_permissions = get_permissions_for_role(token_payload.role)
    
    return AuthenticatedUser(
        user_id=token_payload.sub,
        role=token_payload.role,
        permissions=effective_permissions,
        session_id=token_payload.session_id,
        name=token_payload.name,
        email=token_payload.email,
        token_exp=token_payload.exp,
        token_iss=token_payload.iss,
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[AuthenticatedUser]:
    """
    FastAPI dependency for optionally authenticated endpoints.
    
    Returns None if no token provided, raises error if token is invalid.
    """
    if credentials is None:
        return None
    
    return await get_current_user(credentials)


def require_role(*allowed_roles: UserRole):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))):
            return {"admin": True}
    """
    async def role_checker(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' not authorized. Required: {[r.value for r in allowed_roles]}",
            )
        return user
    
    return role_checker


def require_permission(permission: str):
    """
    Dependency factory for permission-based access control.
    
    Usage:
        @app.get("/reports")
        async def get_reports(user: AuthenticatedUser = Depends(require_permission("view_reports"))):
            return {"reports": [...]}
    """
    async def permission_checker(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return user
    
    return permission_checker


def require_session_access(session_id_param: str = "session_id"):
    """
    Dependency factory to verify user can access a specific interview session.
    
    Usage:
        @app.get("/interview/{session_id}")
        async def get_interview(
            session_id: str,
            user: AuthenticatedUser = Depends(require_session_access("session_id"))
        ):
            return {"session": session_id}
    """
    async def session_checker(
        request: Request,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        # Get session_id from path params
        session_id = request.path_params.get(session_id_param)
        
        if session_id and not user.can_access_session(session_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this interview session",
            )
        return user
    
    return session_checker
