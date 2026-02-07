#!/usr/bin/env python
"""
Pre-Interview Candidate Suitability Analysis.

Analyzes a candidate's resume against a job description to determine:
1. Overall suitability score
2. Skill match analysis
3. Experience match
4. LLM-enhanced insights (strengths, concerns, interview focus areas)
5. Go/No-Go recommendation

Generates both console output and a PDF report.

Usage:
    python scripts/run_pre_interview_analysis.py \
        --resume path/to/resume.pdf \
        --jd path/to/job_description.docx

    # Or with default test files:
    python scripts/run_pre_interview_analysis.py
"""
import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.document_processor import DocumentProcessor
from src.services.semantic_matcher import SemanticMatcher
from src.models.documents import ParsedResume, ParsedJobDescription, MatchResult


# ANSI colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.RESET}\n")


def print_subsection(text: str):
    print(f"\n{Colors.BOLD}{Colors.WHITE}  {text}{Colors.RESET}")


def print_info(label: str, value: str):
    print(f"  {Colors.DIM}{label}:{Colors.RESET} {value}")


def print_score(label: str, score: float, max_score: float = 100):
    """Print a score with a visual bar."""
    percentage = score / max_score
    bar_length = 30
    filled = int(bar_length * percentage)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    if percentage >= 0.7:
        color = Colors.GREEN
    elif percentage >= 0.5:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    print(f"  {label}: {color}{bar} {score:.1f}/{max_score:.0f}{Colors.RESET}")


def print_skills(label: str, skills: list, color: str = Colors.GREEN):
    if skills:
        skills_str = ", ".join(skills[:15])
        if len(skills) > 15:
            skills_str += f" (+{len(skills) - 15} more)"
        print(f"  {label}: {color}{skills_str}{Colors.RESET}")
    else:
        print(f"  {label}: {Colors.DIM}None{Colors.RESET}")


async def load_documents(
    resume_path: Path,
    jd_path: Path,
) -> Tuple[ParsedResume, ParsedJobDescription]:
    """Load and parse resume and JD documents.
    
    Args:
        resume_path: Path to candidate resume file
        jd_path: Path to job description file
    """
    processor = DocumentProcessor()
    
    # Load resume
    print_info("Loading resume", str(resume_path))
    resume_bytes = resume_path.read_bytes()
    
    # Determine content type
    if resume_path.suffix.lower() == ".pdf":
        resume_content_type = "application/pdf"
    elif resume_path.suffix.lower() in [".docx", ".doc"]:
        resume_content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise ValueError(f"Unsupported resume format: {resume_path.suffix}")
    
    print(f"  {Colors.DIM}Using LLM-based parsing (Ollama)...{Colors.RESET}")
    resume_text = await processor.extract_text(resume_bytes, resume_content_type)
    resume = await processor.parse_resume_with_llm(resume_text)
    print(f"  {Colors.GREEN}✓ Resume parsed with LLM successfully{Colors.RESET}")
    
    # Load JD
    print_info("Loading job description", str(jd_path))
    jd_bytes = jd_path.read_bytes()
    
    if jd_path.suffix.lower() == ".pdf":
        jd_text = await processor.extract_text_from_pdf(jd_bytes)
    elif jd_path.suffix.lower() in [".docx", ".doc"]:
        jd_text = await processor.extract_text_from_docx(jd_bytes)
    else:
        # Assume plain text
        jd_text = jd_bytes.decode("utf-8", errors="replace")
    
    print(f"  {Colors.DIM}Using LLM-based parsing (Ollama)...{Colors.RESET}")
    jd = await processor.parse_job_description_with_llm(jd_text)
    print(f"  {Colors.GREEN}✓ Job description parsed with LLM successfully{Colors.RESET}")
    
    return resume, jd


def display_candidate_profile(resume: ParsedResume):
    """Display parsed candidate information."""
    print_section("CANDIDATE PROFILE")
    
    # Contact info
    if resume.contact:
        if resume.contact.name:
            print_info("Name", resume.contact.name)
        if resume.contact.email:
            print_info("Email", resume.contact.email)
        if resume.contact.phone:
            print_info("Phone", resume.contact.phone)
        if resume.contact.linkedin:
            print_info("LinkedIn", resume.contact.linkedin)
        if resume.contact.github:
            print_info("GitHub", resume.contact.github)
    
    # Summary
    if resume.summary:
        print_subsection("Summary")
        summary = resume.summary[:300] + "..." if len(resume.summary) > 300 else resume.summary
        print(f"  {Colors.DIM}{summary}{Colors.RESET}")
    
    # Skills
    print_subsection("Skills")
    print_skills("Technical Skills", resume.skills, Colors.CYAN)
    
    # Experience
    print_subsection("Experience")
    for i, exp in enumerate(resume.experience[:4], 1):
        title = exp.title or "Unknown Role"
        company = exp.company or "Unknown Company"
        dates = ""
        if exp.start_date:
            dates = f" ({exp.start_date}"
            if exp.end_date:
                dates += f" - {exp.end_date})"
            else:
                dates += " - Present)"
        print(f"  {i}. {Colors.WHITE}{title}{Colors.RESET} at {Colors.CYAN}{company}{Colors.RESET}{dates}")
    
    if len(resume.experience) > 4:
        print(f"  {Colors.DIM}... +{len(resume.experience) - 4} more positions{Colors.RESET}")
    
    # Education
    if resume.education:
        print_subsection("Education")
        for edu in resume.education[:2]:
            degree = edu.degree or "Degree"
            institution = edu.institution or "Institution"
            print(f"  • {degree} - {institution}")


def display_job_requirements(jd: ParsedJobDescription):
    """Display parsed job description information."""
    print_section("JOB REQUIREMENTS")
    
    if jd.title:
        print_info("Position", jd.title)
    if jd.company:
        print_info("Company", jd.company)
    if jd.location:
        print_info("Location", jd.location)
    if jd.experience_level:
        print_info("Level", jd.experience_level)
    if jd.experience_years_min:
        exp_range = f"{jd.experience_years_min}+"
        if jd.experience_years_max:
            exp_range = f"{jd.experience_years_min}-{jd.experience_years_max}"
        print_info("Experience Required", f"{exp_range} years")
    
    # Required skills
    print_subsection("Required Skills")
    print_skills("Must Have", jd.required_skills, Colors.YELLOW)
    
    # Preferred skills
    if jd.preferred_skills:
        print_skills("Nice to Have", jd.preferred_skills, Colors.CYAN)
    
    # Responsibilities
    if jd.responsibilities:
        print_subsection("Key Responsibilities")
        for resp in jd.responsibilities[:5]:
            resp_short = resp[:80] + "..." if len(resp) > 80 else resp
            print(f"  • {resp_short}")


def display_match_results(match_result: MatchResult):
    """Display the matching results."""
    print_section("SUITABILITY ANALYSIS")
    
    # Score breakdown
    print_subsection("Score Breakdown")
    print_score("Overall Match", match_result.overall_score)
    print_score("Skill Match", match_result.skill_match_score)
    print_score("Experience Match", match_result.experience_match_score)
    print_score("Semantic Similarity", match_result.semantic_similarity_score)
    
    # Matched skills
    print_subsection("Skill Analysis")
    print_skills("Matched Skills", match_result.matched_skills, Colors.GREEN)
    print_skills("Missing Skills", match_result.missing_skills, Colors.RED)
    
    # Coverage
    if match_result.matched_skills or match_result.missing_skills:
        total = len(match_result.matched_skills) + len(match_result.missing_skills)
        coverage = len(match_result.matched_skills) / total * 100 if total > 0 else 0
        print(f"\n  {Colors.DIM}Skill Coverage: {coverage:.0f}% ({len(match_result.matched_skills)}/{total}){Colors.RESET}")


async def generate_llm_insights(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    match_result: MatchResult,
) -> Optional[dict]:
    """Generate LLM-enhanced insights about candidate fit."""
    print_section("LLM-ENHANCED INSIGHTS")
    print(f"  {Colors.DIM}Generating insights with Ollama...{Colors.RESET}")
    
    try:
        from src.providers.llm import get_llm_provider
        from src.providers.llm.base import GenerationConfig, user_message, system_message
        
        llm = await get_llm_provider()
        
        # Build context
        resume_summary = []
        if resume.summary:
            resume_summary.append(f"Summary: {resume.summary[:500]}")
        if resume.skills:
            resume_summary.append(f"Skills: {', '.join(resume.skills[:20])}")
        for exp in resume.experience[:3]:
            if exp.title and exp.company:
                resume_summary.append(f"Experience: {exp.title} at {exp.company}")
        
        jd_summary = []
        if jd.title:
            jd_summary.append(f"Role: {jd.title}")
        if jd.required_skills:
            jd_summary.append(f"Required Skills: {', '.join(jd.required_skills[:15])}")
        if jd.responsibilities:
            jd_summary.append(f"Responsibilities: {'; '.join(jd.responsibilities[:5])}")
        
        prompt = f"""Analyze this candidate's fit for the job and provide insights.

**Candidate Resume:**
{chr(10).join(resume_summary)}

**Job Description:**
{chr(10).join(jd_summary)}

**Match Scores:**
- Overall: {match_result.overall_score:.1f}/100
- Skill Match: {match_result.skill_match_score:.1f}/100
- Experience Match: {match_result.experience_match_score:.1f}/100

**Matched Skills:** {', '.join(match_result.matched_skills[:10]) or 'None'}
**Missing Skills:** {', '.join(match_result.missing_skills[:10]) or 'None'}

Provide a brief analysis in this exact JSON format:
{{
    "overall_fit": "strong|moderate|weak",
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "concerns": ["concern 1", "concern 2"],
    "interview_focus_areas": ["area 1", "area 2", "area 3"],
    "summary": "2-3 sentence overall assessment"
}}

Return ONLY the JSON, no other text."""

        messages = [
            system_message("You are an expert technical recruiter analyzing candidate fit."),
            user_message(prompt),
        ]
        
        response = await llm.generate(messages, GenerationConfig(max_tokens=1024, temperature=0.3))
        
        # Parse JSON response
        import json
        content = response.content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        # Find JSON object
        if not content.startswith("{"):
            start = content.find("{")
            if start != -1:
                content = content[start:]
        if not content.endswith("}"):
            end = content.rfind("}")
            if end != -1:
                content = content[:end + 1]
        
        insights = json.loads(content)
        return insights
        
    except Exception as e:
        print(f"  {Colors.YELLOW}⚠ LLM analysis failed: {e}{Colors.RESET}")
        return None


def display_llm_insights(insights: dict):
    """Display LLM-generated insights."""
    if not insights:
        return
    
    # Overall fit
    fit = insights.get("overall_fit", "unknown")
    fit_color = Colors.GREEN if fit == "strong" else Colors.YELLOW if fit == "moderate" else Colors.RED
    print(f"\n  {Colors.BOLD}Overall Fit:{Colors.RESET} {fit_color}{fit.upper()}{Colors.RESET}")
    
    # Summary
    if insights.get("summary"):
        print(f"\n  {Colors.DIM}{insights['summary']}{Colors.RESET}")
    
    # Strengths
    if insights.get("strengths"):
        print_subsection("Key Strengths for This Role")
        for s in insights["strengths"][:5]:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {s}")
    
    # Concerns
    if insights.get("concerns"):
        print_subsection("Potential Concerns")
        for c in insights["concerns"][:5]:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} {c}")
    
    # Interview focus
    if insights.get("interview_focus_areas"):
        print_subsection("Suggested Interview Focus Areas")
        for i, area in enumerate(insights["interview_focus_areas"][:5], 1):
            print(f"  {i}. {area}")


def display_recommendation(match_result: MatchResult, insights: Optional[dict]):
    """Display final recommendation."""
    print_section("RECOMMENDATION")
    
    score = match_result.overall_score
    
    # Determine recommendation
    if score >= 70:
        recommendation = "PROCEED TO INTERVIEW"
        color = Colors.GREEN
        emoji = "✓"
        reasoning = "Candidate shows strong alignment with role requirements."
    elif score >= 50:
        recommendation = "CONDITIONAL PROCEED"
        color = Colors.YELLOW
        emoji = "⚠"
        reasoning = "Candidate has potential but may have skill gaps to address."
    else:
        recommendation = "DO NOT PROCEED"
        color = Colors.RED
        emoji = "✗"
        reasoning = "Candidate may not be a good fit for this role."
    
    print(f"\n  {color}{Colors.BOLD}{emoji} {recommendation}{Colors.RESET}")
    print(f"  {Colors.DIM}{reasoning}{Colors.RESET}")
    
    # If we have LLM insights, add them
    if insights and insights.get("overall_fit"):
        fit = insights["overall_fit"]
        if fit == "strong" and score < 70:
            print(f"\n  {Colors.CYAN}Note: LLM analysis suggests stronger fit than scores indicate.{Colors.RESET}")
        elif fit == "weak" and score >= 50:
            print(f"\n  {Colors.YELLOW}Note: LLM analysis suggests concerns despite moderate scores.{Colors.RESET}")


async def generate_pdf_report(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    match_result: MatchResult,
    insights: Optional[dict],
    output_path: Path,
):
    """Generate a PDF report of the pre-interview analysis."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor('#1a365d'),
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#2c5282'),
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#4a5568'),
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("Pre-Interview Candidate Analysis", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Date', parent=body_style, textColor=colors.gray, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 20))
    
    # Candidate Info
    elements.append(Paragraph("Candidate Profile", heading_style))
    candidate_name = resume.contact.name if resume.contact and resume.contact.name else "Unknown Candidate"
    candidate_email = resume.contact.email if resume.contact and resume.contact.email else "N/A"
    elements.append(Paragraph(f"<b>Name:</b> {candidate_name}", body_style))
    elements.append(Paragraph(f"<b>Email:</b> {candidate_email}", body_style))
    if resume.skills:
        skills_str = ", ".join(resume.skills[:15])
        elements.append(Paragraph(f"<b>Key Skills:</b> {skills_str}", body_style))
    
    # Job Info
    elements.append(Paragraph("Position Details", heading_style))
    job_title = jd.title or "Unknown Position"
    job_company = jd.company or "Unknown Company"
    elements.append(Paragraph(f"<b>Role:</b> {job_title}", body_style))
    elements.append(Paragraph(f"<b>Company:</b> {job_company}", body_style))
    if jd.required_skills:
        req_skills = ", ".join(jd.required_skills[:10])
        elements.append(Paragraph(f"<b>Required Skills:</b> {req_skills}", body_style))
    
    # Scores
    elements.append(Paragraph("Suitability Scores", heading_style))
    
    score_data = [
        ["Metric", "Score", "Rating"],
        ["Overall Match", f"{match_result.overall_score:.1f}/100", 
         "Strong" if match_result.overall_score >= 70 else "Moderate" if match_result.overall_score >= 50 else "Weak"],
        ["Skill Match", f"{match_result.skill_match_score:.1f}/100", ""],
        ["Experience Match", f"{match_result.experience_match_score:.1f}/100", ""],
        ["Semantic Similarity", f"{match_result.semantic_similarity_score:.1f}/100", ""],
    ]
    
    score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#e6f3ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 15))
    
    # Skill Analysis
    elements.append(Paragraph("Skill Analysis", heading_style))
    if match_result.matched_skills:
        matched = ", ".join(match_result.matched_skills[:12])
        elements.append(Paragraph(f"<b>Matched Skills:</b> <font color='green'>{matched}</font>", body_style))
    if match_result.missing_skills:
        missing = ", ".join(match_result.missing_skills[:12])
        elements.append(Paragraph(f"<b>Missing Skills:</b> <font color='red'>{missing}</font>", body_style))
    
    # LLM Insights
    if insights:
        elements.append(Paragraph("AI Analysis", heading_style))
        
        fit = insights.get("overall_fit", "unknown")
        fit_color = "green" if fit == "strong" else "orange" if fit == "moderate" else "red"
        elements.append(Paragraph(f"<b>Overall Fit:</b> <font color='{fit_color}'>{fit.upper()}</font>", body_style))
        
        if insights.get("summary"):
            elements.append(Paragraph(f"<i>{insights['summary']}</i>", body_style))
        
        if insights.get("strengths"):
            elements.append(Paragraph("<b>Key Strengths:</b>", subheading_style))
            for s in insights["strengths"][:5]:
                elements.append(Paragraph(f"• {s}", body_style))
        
        if insights.get("concerns"):
            elements.append(Paragraph("<b>Potential Concerns:</b>", subheading_style))
            for c in insights["concerns"][:5]:
                elements.append(Paragraph(f"• {c}", body_style))
        
        if insights.get("interview_focus_areas"):
            elements.append(Paragraph("<b>Suggested Interview Focus:</b>", subheading_style))
            for i, area in enumerate(insights["interview_focus_areas"][:5], 1):
                elements.append(Paragraph(f"{i}. {area}", body_style))
    
    # Recommendation
    elements.append(Paragraph("Recommendation", heading_style))
    score = match_result.overall_score
    if score >= 70:
        rec_text = "<font color='green'><b>PROCEED TO INTERVIEW</b></font>"
        rec_reasoning = "Candidate shows strong alignment with role requirements."
    elif score >= 50:
        rec_text = "<font color='orange'><b>CONDITIONAL PROCEED</b></font>"
        rec_reasoning = "Candidate has potential but may have skill gaps to address."
    else:
        rec_text = "<font color='red'><b>DO NOT PROCEED</b></font>"
        rec_reasoning = "Candidate may not be a good fit for this role."
    
    elements.append(Paragraph(rec_text, body_style))
    elements.append(Paragraph(rec_reasoning, body_style))
    
    # Build PDF
    doc.build(elements)
    return output_path


async def run_analysis(resume_path: Path, jd_path: Path):
    """Run the full pre-interview analysis.
    
    Args:
        resume_path: Path to candidate resume
        jd_path: Path to job description
    """
    start_time = datetime.now()
    
    # Header
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("="*70)
    print("       PRE-INTERVIEW CANDIDATE SUITABILITY ANALYSIS")
    print("="*70)
    print(Colors.RESET)
    
    print(f"  {Colors.DIM}Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"  {Colors.DIM}Parsing mode: LLM-based{Colors.RESET}")
    
    # Step 1: Load documents
    print_header("DOCUMENT PARSING")
    resume, jd = await load_documents(resume_path, jd_path)
    
    # Step 2: Display parsed info
    display_candidate_profile(resume)
    display_job_requirements(jd)
    
    # Step 3: Run semantic matching
    print_header("RUNNING SUITABILITY ANALYSIS")
    print(f"  {Colors.DIM}Loading embedding model...{Colors.RESET}")
    
    matcher = SemanticMatcher()
    print(f"  {Colors.GREEN}✓ Model loaded: {matcher.model_name}{Colors.RESET}")
    
    match_result = await matcher.match(
        resume=resume,
        job_description=jd,
        resume_id="candidate-resume",
        job_description_id="target-jd",
    )
    
    display_match_results(match_result)
    
    # Step 4: Generate LLM insights
    insights = await generate_llm_insights(resume, jd, match_result)
    if insights:
        display_llm_insights(insights)
    
    # Step 5: Display recommendation
    display_recommendation(match_result, insights)
    
    # Step 6: Generate PDF report
    print_header("GENERATING PDF REPORT")
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_name = "unknown"
    if resume.contact and resume.contact.name:
        candidate_name = resume.contact.name.replace(" ", "_").lower()[:20]
    
    pdf_path = reports_dir / f"pre_interview_{candidate_name}_{timestamp}.pdf"
    
    await generate_pdf_report(resume, jd, match_result, insights, pdf_path)
    
    print(f"  {Colors.GREEN}✓ PDF Report saved:{Colors.RESET} {pdf_path}")
    print(f"  {Colors.DIM}File size: {pdf_path.stat().st_size / 1024:.1f} KB{Colors.RESET}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("="*70)
    print("              ANALYSIS COMPLETE")
    print(f"       Duration: {duration:.1f}s")
    print("="*70)
    print(Colors.RESET)
    
    return match_result, insights


def find_default_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Try to find default test files."""
    test_dir = project_root / "tests_personal"
    
    if not test_dir.exists():
        return None, None
    
    resume_path = None
    jd_path = None
    
    for f in test_dir.iterdir():
        if f.suffix.lower() == ".pdf" and "resume" in f.name.lower():
            resume_path = f
        elif f.suffix.lower() in [".docx", ".doc"]:
            jd_path = f
        elif f.suffix.lower() == ".pdf" and resume_path is None:
            resume_path = f
    
    return resume_path, jd_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Interview Candidate Suitability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume", "-r",
        type=Path,
        help="Path to candidate resume (PDF or DOCX)",
    )
    parser.add_argument(
        "--jd", "-j",
        type=Path,
        help="Path to job description (PDF, DOCX, or TXT)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached parsed documents before running",
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        processor = DocumentProcessor()
        processor.clear_cache()
        print(f"{Colors.YELLOW}Cache cleared.{Colors.RESET}")
    
    # Find files
    resume_path = args.resume
    jd_path = args.jd
    
    if not resume_path or not jd_path:
        default_resume, default_jd = find_default_files()
        
        if not resume_path:
            resume_path = default_resume
        if not jd_path:
            jd_path = default_jd
    
    # Validate
    if not resume_path or not resume_path.exists():
        print(f"{Colors.RED}Error: Resume file not found: {resume_path}{Colors.RESET}")
        print(f"Usage: python {__file__} --resume path/to/resume.pdf --jd path/to/jd.docx")
        sys.exit(1)
    
    if not jd_path or not jd_path.exists():
        print(f"{Colors.RED}Error: Job description file not found: {jd_path}{Colors.RESET}")
        print(f"Usage: python {__file__} --resume path/to/resume.pdf --jd path/to/jd.docx")
        sys.exit(1)
    
    # Run analysis
    asyncio.run(run_analysis(resume_path, jd_path))


if __name__ == "__main__":
    main()
