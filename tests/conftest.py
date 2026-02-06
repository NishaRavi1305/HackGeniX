"""
pytest configuration and shared fixtures.
"""
import os
import pytest
import asyncio
from typing import Generator

# Disable auth for API tests by default
os.environ["AUTH_ENABLED"] = "false"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def auth_headers():
    """
    Provides auth headers for API tests when auth is enabled.
    
    Usage:
        response = await client.get("/api/endpoint", headers=auth_headers)
    """
    from src.core.auth import create_token
    from src.models.auth import UserRole
    from datetime import timedelta
    
    token = create_token(
        subject="test-user",
        role=UserRole.ADMIN,
        expires_delta=timedelta(hours=1),
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_for_role():
    """
    Factory fixture to create auth headers for specific roles.
    
    Usage:
        headers = auth_headers_for_role(UserRole.INTERVIEWER)
        response = await client.get("/api/endpoint", headers=headers)
    """
    from src.core.auth import create_token
    from src.models.auth import UserRole
    from datetime import timedelta
    
    def _create_headers(role: UserRole, session_id: str = None):
        token = create_token(
            subject=f"test-{role.value}",
            role=role,
            session_id=session_id,
            expires_delta=timedelta(hours=1),
        )
        return {"Authorization": f"Bearer {token}"}
    
    return _create_headers


# Add any shared fixtures here
