"""
Local filesystem storage provider for PDF reports.

Stores reports in ./reports/{session_id}.pdf by default.
Suitable for development, testing, and single-server deployments.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from src.providers.report_storage.base import ReportStorageProvider, StorageMetadata

logger = logging.getLogger(__name__)


class LocalReportStorage(ReportStorageProvider):
    """
    Store reports in local filesystem.
    
    Default location: ./reports/
    File naming: {session_id}.pdf
    
    Metadata is stored in a companion .meta file (JSON).
    """
    
    def __init__(self, base_path: str = "reports"):
        """
        Initialize local storage provider.
        
        Args:
            base_path: Base directory for storing reports.
                       Relative to current working directory.
        """
        self.base_path = Path(base_path)
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create the reports directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Report storage directory: {self.base_path.absolute()}")
    
    def _get_pdf_path(self, session_id: str) -> Path:
        """Get the path for a session's PDF file."""
        # Sanitize session_id for filesystem safety
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.base_path / f"{safe_id}.pdf"
    
    def _get_meta_path(self, session_id: str) -> Path:
        """Get the path for a session's metadata file."""
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.base_path / f"{safe_id}.meta"
    
    async def store(
        self,
        session_id: str,
        pdf_bytes: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Store a PDF report in the local filesystem.
        
        Args:
            session_id: Unique interview session ID
            pdf_bytes: PDF content as bytes
            metadata: Optional custom metadata to store
            
        Returns:
            Absolute path to the stored PDF
        """
        pdf_path = self._get_pdf_path(session_id)
        meta_path = self._get_meta_path(session_id)
        
        # Write PDF
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        
        # Write metadata
        import json
        meta = {
            "session_id": session_id,
            "stored_at": datetime.utcnow().isoformat(),
            "size_bytes": len(pdf_bytes),
            "content_type": "application/pdf",
            "custom_metadata": metadata or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Stored report: {pdf_path} ({len(pdf_bytes)} bytes)")
        
        return str(pdf_path.absolute())
    
    async def retrieve(self, session_id: str) -> Optional[bytes]:
        """
        Retrieve a stored PDF report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            PDF content as bytes, or None if not found
        """
        pdf_path = self._get_pdf_path(session_id)
        
        if not pdf_path.exists():
            logger.debug(f"Report not found: {pdf_path}")
            return None
        
        with open(pdf_path, "rb") as f:
            return f.read()
    
    async def exists(self, session_id: str) -> bool:
        """
        Check if a report exists for the given session.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            True if report exists, False otherwise
        """
        return self._get_pdf_path(session_id).exists()
    
    async def delete(self, session_id: str) -> bool:
        """
        Delete a stored report and its metadata.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            True if deleted successfully, False if not found
        """
        pdf_path = self._get_pdf_path(session_id)
        meta_path = self._get_meta_path(session_id)
        
        deleted = False
        
        if pdf_path.exists():
            pdf_path.unlink()
            deleted = True
            logger.info(f"Deleted report: {pdf_path}")
        
        if meta_path.exists():
            meta_path.unlink()
        
        return deleted
    
    def get_url(self, session_id: str) -> Optional[str]:
        """
        Get the file path for a stored report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            Absolute path string
        """
        return str(self._get_pdf_path(session_id).absolute())
    
    async def get_metadata(self, session_id: str) -> Optional[StorageMetadata]:
        """
        Get metadata for a stored report.
        
        Args:
            session_id: Interview session ID
            
        Returns:
            StorageMetadata if report exists, None otherwise
        """
        meta_path = self._get_meta_path(session_id)
        
        if not meta_path.exists():
            # Try to get basic metadata from the PDF file itself
            pdf_path = self._get_pdf_path(session_id)
            if pdf_path.exists():
                stat = pdf_path.stat()
                return StorageMetadata(
                    session_id=session_id,
                    stored_at=datetime.fromtimestamp(stat.st_mtime),
                    size_bytes=stat.st_size,
                )
            return None
        
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        return StorageMetadata(
            session_id=meta["session_id"],
            stored_at=datetime.fromisoformat(meta["stored_at"]),
            size_bytes=meta["size_bytes"],
            content_type=meta.get("content_type", "application/pdf"),
            custom_metadata=meta.get("custom_metadata", {}),
        )
    
    async def list_reports(
        self,
        limit: int = 100,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        List stored reports.
        
        Args:
            limit: Maximum number of reports to return
            prefix: Optional prefix filter for session IDs
            
        Returns:
            List of session IDs with stored reports
        """
        reports = []
        
        for pdf_file in self.base_path.glob("*.pdf"):
            session_id = pdf_file.stem  # filename without extension
            
            if prefix and not session_id.startswith(prefix):
                continue
            
            reports.append(session_id)
            
            if len(reports) >= limit:
                break
        
        # Sort by modification time (newest first)
        reports.sort(
            key=lambda s: self._get_pdf_path(s).stat().st_mtime,
            reverse=True,
        )
        
        return reports[:limit]
