"""
Report Storage Providers.

Modular plug-and-play storage backends for interview PDF reports.
Default: LocalReportStorage (stores in ./reports/)
Future: S3ReportStorage for cloud storage
"""
from src.providers.report_storage.base import ReportStorageProvider
from src.providers.report_storage.local_provider import LocalReportStorage

# Default provider instance (lazy loaded)
_default_provider: ReportStorageProvider = None


def get_report_storage_provider() -> ReportStorageProvider:
    """
    Get the configured report storage provider.
    
    Currently defaults to LocalReportStorage.
    Can be extended to read from config and support S3.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = LocalReportStorage()
    return _default_provider


def set_report_storage_provider(provider: ReportStorageProvider):
    """
    Set a custom report storage provider.
    
    Useful for testing or switching to S3 storage.
    """
    global _default_provider
    _default_provider = provider


__all__ = [
    "ReportStorageProvider",
    "LocalReportStorage",
    "get_report_storage_provider",
    "set_report_storage_provider",
]
