"""
S3/MinIO storage client for file uploads with local filesystem fallback.

Handles resume PDFs, audio recordings, and other assets.
Falls back to local filesystem storage when S3/MinIO is unavailable.
"""
import io
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO, Protocol, runtime_checkable
from datetime import datetime
from abc import ABC, abstractmethod

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends."""
    
    async def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        category: str,
        content_type: Optional[str],
        user_id: Optional[str],
    ) -> str: ...
    
    async def upload_bytes(
        self,
        data: bytes,
        filename: str,
        category: str,
        content_type: Optional[str],
        user_id: Optional[str],
    ) -> str: ...
    
    async def download_file(self, key: str) -> bytes: ...
    async def delete_file(self, key: str) -> bool: ...
    async def file_exists(self, key: str) -> bool: ...
    async def health_check(self) -> bool: ...


class LocalStorageClient:
    """
    Local filesystem storage client.
    Used as fallback when S3/MinIO is not available.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self._base_path = Path(base_path or "./storage")
        self._initialized = False
    
    async def initialize(self) -> None:
        """Create the base storage directory."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info(f"Local storage initialized at: {self._base_path.absolute()}")
    
    def _generate_key(self, category: str, filename: str, user_id: Optional[str] = None) -> str:
        """Generate a structured path for the file."""
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        
        if user_id:
            return f"{category}/{user_id}/{date_path}/{filename}"
        return f"{category}/{date_path}/{filename}"
    
    def _get_full_path(self, key: str) -> Path:
        """Get the full filesystem path for a key."""
        return self._base_path / key
    
    async def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Upload a file to local storage."""
        key = self._generate_key(category, filename, user_id)
        full_path = self._get_full_path(key)
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(full_path, "wb") as f:
            shutil.copyfileobj(file_data, f)
        
        logger.info(f"Uploaded file to local storage: {key}")
        return key
    
    async def upload_bytes(
        self,
        data: bytes,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Upload bytes data to local storage."""
        return await self.upload_file(
            io.BytesIO(data),
            filename,
            category,
            content_type,
            user_id,
        )
    
    async def download_file(self, key: str) -> bytes:
        """Download a file from local storage."""
        full_path = self._get_full_path(key)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        
        with open(full_path, "rb") as f:
            return f.read()
    
    async def delete_file(self, key: str) -> bool:
        """Delete a file from local storage."""
        full_path = self._get_full_path(key)
        
        try:
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Deleted file from local storage: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {key}: {e}")
            return False
    
    async def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a 'presigned URL' for local files.
        Returns a file:// URL since there's no real presigning for local files.
        """
        full_path = self._get_full_path(key)
        return f"file://{full_path.absolute()}"
    
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in local storage."""
        return self._get_full_path(key).exists()
    
    async def health_check(self) -> bool:
        """Check if local storage is accessible."""
        return self._initialized and self._base_path.exists()


class S3StorageClient:
    """
    S3-compatible storage client (works with MinIO, AWS S3, etc.)
    """
    
    def __init__(self):
        self._client = None
        self._bucket = settings.s3_bucket_name
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the S3 client and ensure bucket exists.
        """
        import boto3
        from botocore.config import Config
        
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4"),
        )
        
        # Ensure bucket exists
        await self._ensure_bucket_exists()
        self._initialized = True
        logger.info(f"S3 storage client initialized with bucket: {self._bucket}")
    
    async def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        from botocore.exceptions import ClientError
        
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                self._client.create_bucket(Bucket=self._bucket)
                logger.info(f"Created bucket: {self._bucket}")
            else:
                raise
    
    def _generate_key(self, category: str, filename: str, user_id: Optional[str] = None) -> str:
        """
        Generate a structured S3 key for the file.
        
        Structure: category/[user_id/]YYYY/MM/DD/filename
        """
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        
        if user_id:
            return f"{category}/{user_id}/{date_path}/{filename}"
        return f"{category}/{date_path}/{filename}"
    
    async def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Upload a file to S3 storage.
        
        Args:
            file_data: File-like object to upload
            filename: Original filename
            category: Category/folder for the file (resumes, audio, etc.)
            content_type: MIME type of the file
            user_id: Optional user ID for organizing files
            
        Returns:
            The S3 key of the uploaded file
        """
        from botocore.exceptions import ClientError
        
        key = self._generate_key(category, filename, user_id)
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        
        try:
            self._client.upload_fileobj(
                file_data,
                self._bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )
            logger.info(f"Uploaded file to S3: {key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload file {filename}: {e}")
            raise
    
    async def upload_bytes(
        self,
        data: bytes,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Upload bytes data to S3."""
        return await self.upload_file(
            io.BytesIO(data),
            filename,
            category,
            content_type,
            user_id,
        )
    
    async def download_file(self, key: str) -> bytes:
        """
        Download a file from S3 storage.
        
        Args:
            key: S3 key of the file
            
        Returns:
            File contents as bytes
        """
        from botocore.exceptions import ClientError
        
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to download file {key}: {e}")
            raise
    
    async def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3 storage.
        
        Args:
            key: S3 key of the file
            
        Returns:
            True if deletion was successful
        """
        from botocore.exceptions import ClientError
        
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.info(f"Deleted file from S3: {key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file {key}: {e}")
            return False
    
    async def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary file access.
        
        Args:
            key: S3 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL string
        """
        from botocore.exceptions import ClientError
        
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            raise
    
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in storage."""
        from botocore.exceptions import ClientError
        
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False
    
    async def health_check(self) -> bool:
        """Check if storage is accessible."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except Exception:
            return False


class StorageClient:
    """
    Storage client with automatic fallback.
    Tries S3/MinIO first, falls back to local filesystem if unavailable.
    """
    
    def __init__(self):
        self._backend: Optional[S3StorageClient | LocalStorageClient] = None
        self._storage_type: str = "none"
    
    @property
    def storage_type(self) -> str:
        """Get the current storage backend type."""
        return self._storage_type
    
    async def initialize(self) -> None:
        """
        Initialize storage with fallback.
        Tries S3 first, falls back to local storage if S3 is unavailable.
        """
        # Try S3/MinIO first
        try:
            s3_client = S3StorageClient()
            await s3_client.initialize()
            self._backend = s3_client
            self._storage_type = "s3"
            logger.info("Using S3/MinIO storage backend")
            return
        except Exception as e:
            logger.warning(f"S3/MinIO unavailable ({e}), falling back to local storage")
        
        # Fall back to local storage
        try:
            local_client = LocalStorageClient()
            await local_client.initialize()
            self._backend = local_client
            self._storage_type = "local"
            logger.info("Using local filesystem storage backend")
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            raise RuntimeError("No storage backend available")
    
    async def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Upload a file to storage."""
        return await self._backend.upload_file(
            file_data, filename, category, content_type, user_id
        )
    
    async def upload_bytes(
        self,
        data: bytes,
        filename: str,
        category: str = "uploads",
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Upload bytes data to storage."""
        return await self._backend.upload_bytes(
            data, filename, category, content_type, user_id
        )
    
    async def download_file(self, key: str) -> bytes:
        """Download a file from storage."""
        return await self._backend.download_file(key)
    
    async def delete_file(self, key: str) -> bool:
        """Delete a file from storage."""
        return await self._backend.delete_file(key)
    
    async def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for file access."""
        return await self._backend.get_presigned_url(key, expiration)
    
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in storage."""
        return await self._backend.file_exists(key)
    
    async def health_check(self) -> bool:
        """Check if storage is accessible."""
        if self._backend is None:
            return False
        return await self._backend.health_check()


# Global storage client instance
storage_client = StorageClient()


# FastAPI dependency
async def get_storage() -> StorageClient:
    """FastAPI dependency to get storage client."""
    return storage_client
