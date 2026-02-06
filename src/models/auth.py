"""
Authentication models for the AI Interviewer System.

Supports plug-and-play JWT authentication where the client's backend
issues JWTs and our service validates them.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    HIRING_MANAGER = "hiring_manager"
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    
    @classmethod
    def from_string(cls, value: str) -> "UserRole":
        """Convert string to UserRole, with fallback to CANDIDATE."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.CANDIDATE


class TokenPayload(BaseModel):
    """
    JWT token payload structure.
    
    This is the expected structure of JWTs issued by the client's backend.
    The client should issue JWTs with these claims.
    
    Required claims:
        - sub: Subject (user ID or session ID)
        - exp: Expiration timestamp
        
    Optional claims:
        - role: User role (admin, hiring_manager, interviewer, candidate)
        - iat: Issued at timestamp
        - iss: Issuer (client's system identifier)
        - aud: Audience (should be "ai-interviewer" or configurable)
        - session_id: Interview session ID (for candidates)
        - permissions: List of specific permissions (overrides role-based)
    """
    # Required
    sub: str = Field(..., description="Subject - user ID or session ID")
    exp: int = Field(..., description="Expiration timestamp (Unix epoch)")
    
    # Optional - Identity
    role: UserRole = Field(default=UserRole.CANDIDATE, description="User role")
    iat: Optional[int] = Field(default=None, description="Issued at timestamp")
    iss: Optional[str] = Field(default=None, description="Issuer identifier")
    aud: Optional[str] = Field(default=None, description="Audience")
    
    # Optional - Session-based access for candidates
    session_id: Optional[str] = Field(
        default=None, 
        description="Interview session ID (for candidate access)"
    )
    
    # Optional - Fine-grained permissions
    permissions: Optional[List[str]] = Field(
        default=None,
        description="Explicit permissions list (overrides role-based)"
    )
    
    # Optional - Additional claims
    name: Optional[str] = Field(default=None, description="User display name")
    email: Optional[str] = Field(default=None, description="User email")
    
    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.utcnow().timestamp() > self.exp
    
    @property
    def user_id(self) -> str:
        """Alias for sub claim."""
        return self.sub


class AuthenticatedUser(BaseModel):
    """
    Represents an authenticated user/session.
    
    This is injected into route handlers after successful authentication.
    """
    user_id: str = Field(..., description="User ID from token")
    role: UserRole = Field(..., description="User role")
    permissions: List[str] = Field(default_factory=list, description="Effective permissions")
    session_id: Optional[str] = Field(default=None, description="Interview session ID")
    
    # Optional identity info
    name: Optional[str] = None
    email: Optional[str] = None
    
    # Token metadata
    token_exp: int = Field(..., description="Token expiration timestamp")
    token_iss: Optional[str] = None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or "all" in self.permissions
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)
    
    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if user has all specified permissions."""
        return all(self.has_permission(p) for p in permissions)
    
    def can_access_session(self, session_id: str) -> bool:
        """Check if user can access a specific interview session."""
        # Admins and hiring managers can access any session
        if self.role in [UserRole.ADMIN, UserRole.HIRING_MANAGER]:
            return True
        # Candidates can only access their assigned session
        if self.role == UserRole.CANDIDATE:
            return self.session_id == session_id
        # Interviewers - check permissions
        return self.has_permission(f"session:{session_id}")


class AuthConfig(BaseModel):
    """
    Authentication configuration.
    
    This can be loaded from environment variables or config file,
    making the auth system plug-and-play.
    """
    # JWT Settings
    secret_key: str = Field(..., description="JWT secret key (shared with client)")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # Validation settings
    verify_exp: bool = Field(default=True, description="Verify token expiration")
    verify_iss: bool = Field(default=False, description="Verify issuer claim")
    verify_aud: bool = Field(default=False, description="Verify audience claim")
    
    # Expected values (if verification enabled)
    expected_issuer: Optional[str] = Field(default=None, description="Expected issuer")
    expected_audience: Optional[str] = Field(default="ai-interviewer", description="Expected audience")
    
    # Leeway for clock skew (seconds)
    leeway: int = Field(default=30, description="Leeway for exp/iat validation")
    
    # Public endpoints that don't require auth
    public_paths: List[str] = Field(
        default_factory=lambda: [
            "/health",
            "/api/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
        description="Paths that don't require authentication"
    )


class TokenResponse(BaseModel):
    """Response model for token generation (if we provide it)."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")
