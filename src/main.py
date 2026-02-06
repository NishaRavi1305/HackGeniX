"""
AI Interviewer System - Main FastAPI Application

This is the entry point for the interview system API.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.core.database import mongodb_client
from src.core.storage import storage_client
from src.api import health, documents, interviews, questions, voice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Interviewer System...")
    
    # Initialize MongoDB connection
    await mongodb_client.connect()
    logger.info("MongoDB connection established")
    
    # Initialize S3/MinIO storage
    await storage_client.initialize()
    logger.info("Storage client initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Interviewer System...")
    await mongodb_client.disconnect()
    logger.info("MongoDB connection closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered autonomous interview system for technical and behavioral interviews",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(interviews.router, prefix="/api/v1/interviews", tags=["Interviews"])
    app.include_router(questions.router, prefix="/api/v1/questions", tags=["Questions"])
    app.include_router(voice.router, tags=["Voice"])
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
