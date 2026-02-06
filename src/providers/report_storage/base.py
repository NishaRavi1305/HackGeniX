"""
Abstract base class for report storage providers.

Defines the interface that all storage backends must implement.
This enables plug-and-play switching between local, S3, or other storage.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StorageMetadata:
    """Metadata for a stored report."""
    session_id: str
    stored_at: datetime
    size_bytes: int
    content_type: str = "application/pdf"
    custom_metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_metadata is None:
            self.custom_metadata = {}


class ReportStorageProvider(ABC):
    """
    Abstract interface for report storage backends.
    
    Implementations:
    - LocalReportStorage: Stores in local filesystem (./reports/)
    - S3ReportStorage: Stores in AWS S3 or MinIO (future)
    
    Usage:
        provider = get_report_storage_provider()
        
        # Store a report
        path = await provider.store("session_123", pdf_bytes)
        
        # Retrieve a report
        pdf_bytes = await provider.retrieve("session_123")
        
        # Check existence
        if await provider.exists("session_123"):
            url = provider.get_url("session_123")
    """
    
    @abstractmethod
    async def store(
        self,
        session_id: str,
        pdf_bytes: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Store a PDF report.
        
        Args:
            session_id: Unique interview session ID
            pdf_bytes: PDF content as bytes
            metadata: Optional custom metadata to store
            
        Returns:
            URL or path where the report is stored
        """
        pass
    
    @abstractmethod
    async def retrieve(self, session_id: str) -> Optional[bytes]:
        """
        Retrieve a stored PDF report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            PDF content as bytes, or None if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """
        Check if a report exists for the given session.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            True if report exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete a stored report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            True if deleted successfully, False if not found
        """
        pass
    
    @abstractmethod
    def get_url(self, session_id: str) -> Optional[str]:
        """
        Get the URL or path for a stored report.
        
        This may be a file path (local) or HTTP URL (S3).
        Does not check if the report actually exists.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            URL or path string
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, session_id: str) -> Optional[StorageMetadata]:
        """
        Get metadata for a stored report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            StorageMetadata if report exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_reports(
        self,
        limit: int = 100,
        prefix: Optional[str] = None,
    ) -> list:
        """
        List stored reports.
        
        Args:
            limit: Maximum number of reports to return
            prefix: Optional prefix filter for session IDs
            
        Returns:
            List of session IDs with stored reports
        """
        pass
