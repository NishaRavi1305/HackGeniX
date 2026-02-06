"""
S3/MinIO storage provider for PDF reports.

This is a stub implementation for future cloud storage support.
When implemented, this will store reports in S3-compatible storage.

To use S3 storage:
1. Install boto3: pip install boto3
2. Configure AWS credentials
3. Implement the methods below
4. Update config to use S3ReportStorage
"""
import logging
from typing import Optional, Dict, List

from src.providers.report_storage.base import ReportStorageProvider, StorageMetadata

logger = logging.getLogger(__name__)


class S3ReportStorage(ReportStorageProvider):
    """
    Store reports in AWS S3 or S3-compatible storage (MinIO).
    
    NOT YET IMPLEMENTED - This is a placeholder for future development.
    
    Configuration (via environment variables):
    - AWS_ACCESS_KEY_ID: AWS access key
    - AWS_SECRET_ACCESS_KEY: AWS secret key
    - AWS_REGION: AWS region (default: us-east-1)
    - S3_BUCKET: Bucket name for reports
    - S3_PREFIX: Key prefix (default: reports/)
    - S3_ENDPOINT_URL: Custom endpoint for MinIO (optional)
    
    Usage (future):
        provider = S3ReportStorage(
            bucket="my-interview-reports",
            prefix="reports/",
            endpoint_url="http://localhost:9000",  # For MinIO
        )
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "reports/",
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """
        Initialize S3 storage provider.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all reports
            endpoint_url: Custom endpoint URL (for MinIO)
            region: AWS region
        
        Raises:
            NotImplementedError: Always raised - S3 storage not yet implemented
        """
        raise NotImplementedError(
            "S3 storage is not yet implemented. "
            "Use LocalReportStorage for now, or implement this class.\n"
            "\n"
            "To implement S3 storage:\n"
            "1. Install boto3: pip install boto3\n"
            "2. Create S3 client with credentials\n"
            "3. Implement store/retrieve/delete methods using s3.put_object/get_object/delete_object\n"
            "4. Handle pre-signed URLs for get_url()\n"
            "\n"
            "Example implementation sketch:\n"
            "  import boto3\n"
            "  self.s3 = boto3.client('s3', endpoint_url=endpoint_url)\n"
            "  self.bucket = bucket\n"
            "  self.prefix = prefix\n"
        )
        
        # When implemented, initialize like this:
        # import boto3
        # self.bucket = bucket
        # self.prefix = prefix
        # self.s3 = boto3.client(
        #     's3',
        #     endpoint_url=endpoint_url,
        #     region_name=region,
        # )
    
    def _get_key(self, session_id: str) -> str:
        """Get the S3 key for a session's PDF file."""
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return f"{self.prefix}{safe_id}.pdf"
    
    async def store(
        self,
        session_id: str,
        pdf_bytes: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Store a PDF report in S3."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # key = self._get_key(session_id)
        # self.s3.put_object(
        #     Bucket=self.bucket,
        #     Key=key,
        #     Body=pdf_bytes,
        #     ContentType='application/pdf',
        #     Metadata=metadata or {},
        # )
        # return f"s3://{self.bucket}/{key}"
    
    async def retrieve(self, session_id: str) -> Optional[bytes]:
        """Retrieve a stored PDF report from S3."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # try:
        #     response = self.s3.get_object(Bucket=self.bucket, Key=self._get_key(session_id))
        #     return response['Body'].read()
        # except self.s3.exceptions.NoSuchKey:
        #     return None
    
    async def exists(self, session_id: str) -> bool:
        """Check if a report exists in S3."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # try:
        #     self.s3.head_object(Bucket=self.bucket, Key=self._get_key(session_id))
        #     return True
        # except self.s3.exceptions.ClientError:
        #     return False
    
    async def delete(self, session_id: str) -> bool:
        """Delete a stored report from S3."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # try:
        #     self.s3.delete_object(Bucket=self.bucket, Key=self._get_key(session_id))
        #     return True
        # except Exception:
        #     return False
    
    def get_url(self, session_id: str) -> Optional[str]:
        """Get a pre-signed URL for a stored report."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # return self.s3.generate_presigned_url(
        #     'get_object',
        #     Params={'Bucket': self.bucket, 'Key': self._get_key(session_id)},
        #     ExpiresIn=3600,  # 1 hour
        # )
    
    async def get_metadata(self, session_id: str) -> Optional[StorageMetadata]:
        """Get metadata for a stored report."""
        raise NotImplementedError("S3 storage not implemented")
    
    async def list_reports(
        self,
        limit: int = 100,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """List stored reports in S3."""
        raise NotImplementedError("S3 storage not implemented")
        
        # Implementation would be:
        # search_prefix = self.prefix
        # if prefix:
        #     search_prefix += prefix
        # 
        # response = self.s3.list_objects_v2(
        #     Bucket=self.bucket,
        #     Prefix=search_prefix,
        #     MaxKeys=limit,
        # )
        # 
        # return [
        #     obj['Key'].replace(self.prefix, '').replace('.pdf', '')
        #     for obj in response.get('Contents', [])
        # ]
