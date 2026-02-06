"""
Report API endpoints.

Provides REST API for retrieving interview reports in JSON and PDF formats.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import io

from src.models.interview import InterviewReportResponse
from src.services.interview_orchestrator import get_interview_orchestrator
from src.providers.report_storage import get_report_storage_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/{session_id}", response_model=InterviewReportResponse)
async def get_report(session_id: str):
    """
    Get the interview report as JSON.
    
    Returns a comprehensive report with scores, strengths, concerns,
    and hiring recommendation.
    
    Args:
        session_id: Interview session ID
        
    Returns:
        InterviewReportResponse with full report data
    """
    orchestrator = get_interview_orchestrator()
    
    try:
        return await orchestrator.generate_report(session_id)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to generate report for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}",
        )


@router.get("/{session_id}/pdf")
async def download_pdf(session_id: str):
    """
    Download the interview report as a PDF file.
    
    Generates a professional PDF report with:
    - Executive summary
    - Score breakdown with visual bars
    - Strengths and areas for improvement
    - Question-by-question breakdown
    - Hiring recommendation
    
    The PDF is generated on-demand and optionally saved to storage.
    
    Args:
        session_id: Interview session ID
        
    Returns:
        StreamingResponse with PDF content
        Content-Disposition: attachment
    """
    orchestrator = get_interview_orchestrator()
    storage = get_report_storage_provider()
    
    try:
        # Check if we have a cached PDF
        if await storage.exists(session_id):
            pdf_bytes = await storage.retrieve(session_id)
            if pdf_bytes:
                logger.info(f"Serving cached PDF for session {session_id}")
                return StreamingResponse(
                    io.BytesIO(pdf_bytes),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=interview_report_{session_id}.pdf",
                        "Content-Length": str(len(pdf_bytes)),
                    },
                )
        
        # Generate new PDF
        pdf_bytes = await orchestrator.export_pdf(session_id)
        
        # Store for future retrieval
        await storage.store(session_id, pdf_bytes, metadata={
            "source": "api_download",
        })
        
        logger.info(f"Generated and stored PDF for session {session_id} ({len(pdf_bytes)} bytes)")
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=interview_report_{session_id}.pdf",
                "Content-Length": str(len(pdf_bytes)),
            },
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}",
        )


@router.get("/{session_id}/pdf/preview")
async def preview_pdf(session_id: str):
    """
    Preview the interview report PDF in browser.
    
    Same as download_pdf but with inline disposition,
    allowing the browser to display the PDF directly.
    
    Args:
        session_id: Interview session ID
        
    Returns:
        StreamingResponse with PDF content
        Content-Disposition: inline
    """
    orchestrator = get_interview_orchestrator()
    storage = get_report_storage_provider()
    
    try:
        # Check if we have a cached PDF
        if await storage.exists(session_id):
            pdf_bytes = await storage.retrieve(session_id)
            if pdf_bytes:
                return StreamingResponse(
                    io.BytesIO(pdf_bytes),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"inline; filename=interview_report_{session_id}.pdf",
                        "Content-Length": str(len(pdf_bytes)),
                    },
                )
        
        # Generate new PDF
        pdf_bytes = await orchestrator.export_pdf(session_id)
        
        # Store for future retrieval
        await storage.store(session_id, pdf_bytes)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename=interview_report_{session_id}.pdf",
                "Content-Length": str(len(pdf_bytes)),
            },
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF preview for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}",
        )


@router.delete("/{session_id}/pdf")
async def delete_pdf(session_id: str):
    """
    Delete a stored PDF report.
    
    Removes the cached PDF from storage. The PDF can be regenerated
    by calling the download endpoint again.
    
    Args:
        session_id: Interview session ID
        
    Returns:
        Success message
    """
    storage = get_report_storage_provider()
    
    if await storage.delete(session_id):
        return {"message": f"PDF report for session {session_id} deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No PDF report found for session {session_id}",
        )


@router.get("/")
async def list_reports(limit: int = 50):
    """
    List all stored PDF reports.
    
    Args:
        limit: Maximum number of reports to return (default: 50)
        
    Returns:
        List of session IDs with stored reports
    """
    storage = get_report_storage_provider()
    
    reports = await storage.list_reports(limit=limit)
    
    # Get metadata for each report
    report_list = []
    for session_id in reports:
        metadata = await storage.get_metadata(session_id)
        if metadata:
            report_list.append({
                "session_id": session_id,
                "stored_at": metadata.stored_at.isoformat(),
                "size_bytes": metadata.size_bytes,
                "url": storage.get_url(session_id),
            })
        else:
            report_list.append({
                "session_id": session_id,
                "url": storage.get_url(session_id),
            })
    
    return {
        "reports": report_list,
        "count": len(report_list),
    }
