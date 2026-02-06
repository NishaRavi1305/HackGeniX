"""
Health check endpoints.
"""
from fastapi import APIRouter, status
from pydantic import BaseModel

from src.core.database import mongodb_client
from src.core.storage import storage_client

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    mongodb: bool
    storage: bool
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all system components.
    """
    mongodb_ok = await mongodb_client.health_check()
    storage_ok = await storage_client.health_check()
    
    overall_status = "healthy" if (mongodb_ok and storage_ok) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        mongodb=mongodb_ok,
        storage=storage_ok,
    )


@router.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AI Interviewer System",
        "version": "0.1.0",
        "docs": "/docs",
    }
