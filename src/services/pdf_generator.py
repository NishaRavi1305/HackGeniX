"""
PDF Report Generator Service.

Generates professional interview assessment PDF reports using ReportLab.
Design: Single long scrollable page with executive summary + full Q&A text.
"""
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)

from src.models.report import (
    FullInterviewReport,
    ScoreSection,
    QuestionSummary,
    ReportStrength,
    ReportConcern,
    RecommendationDecision,
)

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    Generates interview assessment PDF reports.
    
    Design principles:
    - Single long scrollable page (no artificial page breaks)
    - Executive summary followed by full question/answer text
    - Text-based score bars for readability
    - Clean, professional styling without branding
    """
    
    # Score bar characters
    FILLED_BLOCK = "\u2588"  # █
    EMPTY_BLOCK = "\u2591"   # ░
    
    # Color palette
    COLORS = {
        "primary": colors.HexColor("#1a1a2e"),
        "secondary": colors.HexColor("#16213e"),
        "accent": colors.HexColor("#0f3460"),
        "success": colors.HexColor("#22c55e"),
        "warning": colors.HexColor("#f59e0b"),
        "danger": colors.HexColor("#ef4444"),
        "text": colors.HexColor("#1f2937"),
        "text_light": colors.HexColor("#6b7280"),
        "border": colors.HexColor("#e5e7eb"),
        "background": colors.HexColor("#f9fafb"),
    }
    
    def __init__(self, report_data: FullInterviewReport):
        """
        Initialize the PDF generator.
        
        Args:
            report_data: Complete interview report data
        """
        self.data = report_data
        self.styles = self._create_styles()
        self._buffer: Optional[io.BytesIO] = None
    
    def _create_styles(self) -> dict:
        """Create custom paragraph styles."""
        base_styles = getSampleStyleSheet()
        custom_styles = {}
        
        # Title style
        custom_styles["Title"] = ParagraphStyle(
            name="CustomTitle",
            parent=base_styles["Heading1"],
            fontSize=22,
            textColor=self.COLORS["primary"],
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        
        # Subtitle style
        custom_styles["Subtitle"] = ParagraphStyle(
            name="Subtitle",
            parent=base_styles["Normal"],
            fontSize=11,
            textColor=self.COLORS["text_light"],
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        
        # Section header style
        custom_styles["SectionHeader"] = ParagraphStyle(
            name="SectionHeader",
            parent=base_styles["Heading2"],
            fontSize=14,
            textColor=self.COLORS["primary"],
            spaceBefore=20,
            spaceAfter=12,
            fontName="Helvetica-Bold",
            borderPadding=5,
        )
        
        # Subsection header
        custom_styles["SubsectionHeader"] = ParagraphStyle(
            name="SubsectionHeader",
            parent=base_styles["Heading3"],
            fontSize=12,
            textColor=self.COLORS["secondary"],
            spaceBefore=12,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        )
        
        # Body text
        custom_styles["Body"] = ParagraphStyle(
            name="Body",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=self.COLORS["text"],
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14,
        )
        
        # Quote/answer text
        custom_styles["Quote"] = ParagraphStyle(
            name="Quote",
            parent=base_styles["Normal"],
            fontSize=9,
            textColor=self.COLORS["text"],
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20,
            alignment=TA_LEFT,
            leading=13,
            backColor=self.COLORS["background"],
        )
        
        # Bullet point style
        custom_styles["Bullet"] = ParagraphStyle(
            name="Bullet",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=self.COLORS["text"],
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10,
        )
        
        # Score display (large)
        custom_styles["ScoreLarge"] = ParagraphStyle(
            name="ScoreLarge",
            parent=base_styles["Normal"],
            fontSize=28,
            textColor=self.COLORS["primary"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        
        # Score bar text
        custom_styles["ScoreBar"] = ParagraphStyle(
            name="ScoreBar",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=self.COLORS["text"],
            fontName="Courier",
            spaceAfter=4,
        )
        
        # Footer style
        custom_styles["Footer"] = ParagraphStyle(
            name="Footer",
            parent=base_styles["Normal"],
            fontSize=8,
            textColor=self.COLORS["text_light"],
            alignment=TA_CENTER,
        )
        
        # Recommendation badge
        custom_styles["RecommendationBadge"] = ParagraphStyle(
            name="RecommendationBadge",
            parent=base_styles["Normal"],
            fontSize=14,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        
        return custom_styles
    
    def _draw_score_bar(
        self,
        score: float,
        max_score: float = 100.0,
        width: int = 20,
    ) -> str:
        """
        Generate a text-based score bar.
        
        Args:
            score: Current score value
            max_score: Maximum possible score
            width: Number of characters for the bar
            
        Returns:
            String like "████████░░ 80/100"
        """
        if max_score == 0:
            percentage = 0
        else:
            percentage = score / max_score
        
        filled = int(percentage * width)
        empty = width - filled
        
        bar = self.FILLED_BLOCK * filled + self.EMPTY_BLOCK * empty
        return f"{bar} {score:.0f}/{max_score:.0f}"
    
    def _get_score_color(self, score: float, max_score: float = 100.0) -> colors.Color:
        """Get color based on score percentage."""
        if max_score == 0:
            return self.COLORS["text_light"]
        
        percentage = (score / max_score) * 100
        
        if percentage >= 80:
            return self.COLORS["success"]
        elif percentage >= 60:
            return colors.HexColor("#84cc16")  # lime
        elif percentage >= 40:
            return self.COLORS["warning"]
        else:
            return self.COLORS["danger"]
    
    def _build_header(self) -> List:
        """Build the report header section."""
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Interview Assessment Report",
            self.styles["Title"]
        ))
        
        # Subtitle with date
        generated_str = self.data.metadata.generated_at.strftime("%B %d, %Y at %H:%M UTC")
        elements.append(Paragraph(
            f"Generated: {generated_str}",
            self.styles["Subtitle"]
        ))
        
        return elements
    
    def _build_candidate_info(self) -> List:
        """Build the candidate information section."""
        elements = []
        
        candidate = self.data.candidate
        interview_date = candidate.interview_date.strftime("%B %d, %Y")
        
        # Create info table
        data = [
            ["Candidate:", candidate.name, "Position:", candidate.role_title],
            ["Interview Date:", interview_date, "Duration:", f"{candidate.duration_minutes} minutes"],
        ]
        
        table = Table(data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONT", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONT", (1, 0), (1, -1), "Helvetica"),
            ("FONT", (3, 0), (3, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, -1), self.COLORS["text"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, -1), self.COLORS["background"]),
            ("BOX", (0, 0), (-1, -1), 1, self.COLORS["border"]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_executive_summary(self) -> List:
        """Build the executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        elements.append(Paragraph(
            self.data.executive_summary or "No executive summary available.",
            self.styles["Body"]
        ))
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_overall_score(self) -> List:
        """Build the overall score display with recommendation."""
        elements = []
        
        overall = self.data.scores.overall
        recommendation = self.data.recommendation
        
        # Overall score in large text
        score_color = self._get_score_color(overall.score, overall.max_score)
        
        # Create a table for score + recommendation side by side
        score_text = f"<font size='28' color='{score_color.hexval()}'><b>{overall.score:.0f}</b></font><font size='14'>/{overall.max_score:.0f}</font>"
        
        # Recommendation badge color
        badge_color = recommendation.decision_color
        badge_text = recommendation.decision_display
        
        score_para = Paragraph(score_text, self.styles["Body"])
        
        # Rating text
        rating_text = f"<font color='{self.COLORS['text_light'].hexval()}'>{overall.rating}</font>"
        rating_para = Paragraph(rating_text, self.styles["Body"])
        
        # Build score box
        score_data = [
            [Paragraph("<b>Overall Score</b>", self.styles["Body"])],
            [score_para],
            [rating_para],
        ]
        
        score_table = Table(score_data, colWidths=[2.5*inch])
        score_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 2, self.COLORS["border"]),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        
        # Recommendation box
        rec_data = [
            [Paragraph("<b>Recommendation</b>", self.styles["Body"])],
            [Paragraph(f"<font color='{badge_color}'><b>{badge_text}</b></font>", 
                      ParagraphStyle(name="rec", fontSize=16, alignment=TA_CENTER))],
            [Paragraph(f"Confidence: {recommendation.confidence_percent:.0f}%", self.styles["Body"])],
        ]
        
        rec_table = Table(rec_data, colWidths=[2.5*inch])
        rec_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 2, colors.HexColor(badge_color)),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        
        # Combine in a row
        main_table = Table([[score_table, Spacer(0.5*inch, 0), rec_table]], 
                          colWidths=[2.5*inch, 0.5*inch, 2.5*inch])
        main_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        
        elements.append(main_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_score_breakdown(self) -> List:
        """Build the detailed score breakdown with text bars."""
        elements = []
        
        elements.append(Paragraph("Score Breakdown", self.styles["SectionHeader"]))
        
        # Build score rows
        scores = [
            self.data.scores.technical,
            self.data.scores.behavioral,
            self.data.scores.communication,
        ]
        if self.data.scores.problem_solving:
            scores.append(self.data.scores.problem_solving)
        
        score_rows = []
        for score in scores:
            bar = self._draw_score_bar(score.score, score.max_score, width=20)
            color = self._get_score_color(score.score, score.max_score)
            
            # Create row: Category | Bar | Rating
            row = [
                Paragraph(f"<b>{score.category}</b>", self.styles["Body"]),
                Paragraph(f"<font face='Courier' color='{color.hexval()}'>{bar}</font>", 
                         self.styles["ScoreBar"]),
                Paragraph(score.rating, self.styles["Body"]),
            ]
            score_rows.append(row)
        
        score_table = Table(score_rows, colWidths=[1.5*inch, 3.5*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, self.COLORS["border"]),
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_strengths_section(self) -> List:
        """Build the strengths section."""
        elements = []
        
        if not self.data.strengths:
            return elements
        
        elements.append(Paragraph("Key Strengths", self.styles["SectionHeader"]))
        
        for strength in self.data.strengths:
            bullet_text = f"<bullet>&bull;</bullet> <b>{strength.title}</b>: {strength.evidence}"
            elements.append(Paragraph(bullet_text, self.styles["Bullet"]))
        
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_concerns_section(self) -> List:
        """Build the areas for improvement section."""
        elements = []
        
        if not self.data.concerns:
            return elements
        
        elements.append(Paragraph("Areas for Improvement", self.styles["SectionHeader"]))
        
        for concern in self.data.concerns:
            severity_color = {
                "low": self.COLORS["text_light"],
                "medium": self.COLORS["warning"],
                "high": self.COLORS["danger"],
                "critical": self.COLORS["danger"],
            }.get(concern.severity.value, self.COLORS["text"])
            
            bullet_text = f"<bullet>&bull;</bullet> <b>{concern.title}</b> "
            bullet_text += f"<font color='{severity_color.hexval()}'>({concern.severity.value})</font>: "
            bullet_text += concern.evidence
            
            if concern.suggestion:
                bullet_text += f" <i>Suggestion: {concern.suggestion}</i>"
            
            elements.append(Paragraph(bullet_text, self.styles["Bullet"]))
        
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_question_breakdown(self) -> List:
        """Build the question-by-question breakdown with executive summary + full text."""
        elements = []
        
        if not self.data.question_evaluations:
            return elements
        
        elements.append(Paragraph("Question-by-Question Breakdown", self.styles["SectionHeader"]))
        
        # Group questions by stage
        stages = {}
        for q in self.data.question_evaluations:
            stage = q.stage.replace("_", " ").title()
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(q)
        
        for stage_name, questions in stages.items():
            # Stage header
            elements.append(Paragraph(stage_name, self.styles["SubsectionHeader"]))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=self.COLORS["border"]))
            
            for i, q in enumerate(questions, 1):
                # Keep question block together if possible
                question_elements = []
                
                # Question header with score bar
                score_bar = self._draw_score_bar(q.score, q.max_score, width=15)
                score_color = self._get_score_color(q.score, q.max_score)
                
                header_text = f"<b>Q{i}.</b> <font color='{score_color.hexval()}'>{score_bar}</font>"
                question_elements.append(Paragraph(header_text, self.styles["Body"]))
                
                # Executive summary
                question_elements.append(Paragraph(
                    f"<i>{q.executive_summary}</i>",
                    self.styles["Body"]
                ))
                
                # Full question text
                question_elements.append(Paragraph(
                    f"<b>Question:</b> {self._escape_xml(q.question_text)}",
                    self.styles["Body"]
                ))
                
                # Full answer text
                question_elements.append(Paragraph(
                    f"<b>Answer:</b>",
                    self.styles["Body"]
                ))
                question_elements.append(Paragraph(
                    self._escape_xml(q.answer_text) if q.answer_text else "<i>No answer provided</i>",
                    self.styles["Quote"]
                ))
                
                # Strengths and improvements (if any)
                if q.strengths:
                    strengths_text = ", ".join(q.strengths[:3])
                    question_elements.append(Paragraph(
                        f"<font color='{self.COLORS['success'].hexval()}'>Strengths:</font> {strengths_text}",
                        self.styles["Body"]
                    ))
                
                if q.improvements:
                    improvements_text = ", ".join(q.improvements[:3])
                    question_elements.append(Paragraph(
                        f"<font color='{self.COLORS['warning'].hexval()}'>Improvements:</font> {improvements_text}",
                        self.styles["Body"]
                    ))
                
                question_elements.append(Spacer(1, 0.1*inch))
                
                # Try to keep together, but allow breaking for long answers
                elements.append(KeepTogether(question_elements[:4]))  # Keep header + summary together
                elements.extend(question_elements[4:])  # Allow Q&A to break if needed
            
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_recommendation_section(self) -> List:
        """Build the final recommendation section."""
        elements = []
        
        rec = self.data.recommendation
        
        elements.append(Paragraph("Hiring Recommendation", self.styles["SectionHeader"]))
        
        # Recommendation box
        badge_color = rec.decision_color
        
        rec_content = [
            [Paragraph(f"<font size='14' color='{badge_color}'><b>{rec.decision_display}</b></font>", 
                      ParagraphStyle(name="rec", alignment=TA_CENTER, fontSize=14))],
            [Paragraph(f"<b>Confidence:</b> {rec.confidence_percent:.0f}%", self.styles["Body"])],
            [Paragraph(f"<b>Reasoning:</b> {rec.reasoning}", self.styles["Body"])],
        ]
        
        rec_table = Table(rec_content, colWidths=[6.5*inch])
        rec_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), self.COLORS["background"]),
            ("BOX", (0, 0), (-1, -1), 2, colors.HexColor(badge_color)),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 15),
            ("RIGHTPADDING", (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(rec_table)
        
        # Next steps
        if rec.next_steps:
            elements.append(Spacer(1, 0.15*inch))
            elements.append(Paragraph("<b>Recommended Next Steps:</b>", self.styles["Body"]))
            for step in rec.next_steps:
                elements.append(Paragraph(f"<bullet>&bull;</bullet> {step}", self.styles["Bullet"]))
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_footer(self) -> List:
        """Build the report footer."""
        elements = []
        
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 0.1*inch))
        
        footer_text = (
            f"Report ID: {self.data.metadata.session_id} | "
            f"Generated: {self.data.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
            f"Version: {self.data.metadata.report_version}"
        )
        elements.append(Paragraph(footer_text, self.styles["Footer"]))
        
        return elements
    
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters for ReportLab."""
        if not text:
            return ""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
    
    def generate(self) -> bytes:
        """
        Generate the PDF report.
        
        Returns:
            PDF content as bytes
        """
        self._buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            self._buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Build story (content)
        story = []
        
        # Add all sections
        story.extend(self._build_header())
        story.extend(self._build_candidate_info())
        story.extend(self._build_executive_summary())
        story.extend(self._build_overall_score())
        story.extend(self._build_score_breakdown())
        story.extend(self._build_strengths_section())
        story.extend(self._build_concerns_section())
        story.extend(self._build_question_breakdown())
        story.extend(self._build_recommendation_section())
        story.extend(self._build_footer())
        
        # Build PDF
        doc.build(story)
        
        # Get bytes
        pdf_bytes = self._buffer.getvalue()
        self._buffer.close()
        
        logger.info(f"Generated PDF report: {len(pdf_bytes)} bytes")
        
        return pdf_bytes
    
    def save(self, output_path: str) -> str:
        """
        Generate and save the PDF to a file.
        
        Args:
            output_path: Path to save the PDF
            
        Returns:
            Absolute path to saved file
        """
        pdf_bytes = self.generate()
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            f.write(pdf_bytes)
        
        logger.info(f"Saved PDF report to: {path.absolute()}")
        
        return str(path.absolute())


def generate_interview_pdf(report_data: FullInterviewReport) -> bytes:
    """
    Convenience function to generate PDF from report data.
    
    Args:
        report_data: Complete interview report data
        
    Returns:
        PDF content as bytes
    """
    generator = PDFReportGenerator(report_data)
    return generator.generate()


def save_interview_pdf(report_data: FullInterviewReport, output_path: str) -> str:
    """
    Convenience function to generate and save PDF.
    
    Args:
        report_data: Complete interview report data
        output_path: Path to save the PDF
        
    Returns:
        Absolute path to saved file
    """
    generator = PDFReportGenerator(report_data)
    return generator.save(output_path)
