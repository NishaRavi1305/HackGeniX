"""
S3/MinIO storage client for file uploads.

Handles resume PDFs, audio recordings, and other assets.
"""
import io
import logging
from typing import Optional, BinaryIO
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageClient:
    """
    S3-compatible storage client (works with MinIO, AWS S3, etc.)
    """
    
    def __init__(self):
        self._client = None
        self._bucket = settings.s3_bucket_name
    
    async def initialize(self) -> None:
        """
        Initialize the S3 client and ensure bucket exists.
        """
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
        logger.info(f"Storage client initialized with bucket: {self._bucket}")
    
    async def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
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
            logger.info(f"Uploaded file: {key}")
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
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.info(f"Deleted file: {key}")
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


# Global storage client instance
storage_client = StorageClient()


# FastAPI dependency
async def get_storage() -> StorageClient:
    """FastAPI dependency to get storage client."""
    return storage_client
