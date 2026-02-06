"""
Document processing service with spaCy NER.

Handles PDF/DOCX parsing and named entity extraction for resumes and job descriptions.
"""
import io
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import spacy
from pdfplumber import open as open_pdf
from docx import Document as DocxDocument

from src.models.documents import (
    ParsedResume,
    ParsedJobDescription,
    ParsedEntity,
    ContactInfo,
    Education,
    Experience,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processing service for resumes and job descriptions.
    
    Uses spaCy for NER and custom patterns for structured extraction.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the document processor.
        
        Args:
            spacy_model: Name of the spaCy model to load
        """
        logger.info(f"Loading spaCy model: {spacy_model}")
        self.nlp = spacy.load(spacy_model)
        
        # Common skill patterns
        self._skill_patterns = self._build_skill_patterns()
        
        # Email regex
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone regex (various formats)
        self._phone_pattern = re.compile(
            r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        )
        
        # LinkedIn URL pattern
        self._linkedin_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        )
        
        # GitHub URL pattern
        self._github_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
        )
        
        # Date patterns
        self._date_pattern = re.compile(
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s*\d{4}|'
            r'\d{1,2}/\d{4}|'
            r'\d{4}\s*[-–]\s*(?:\d{4}|[Pp]resent|[Cc]urrent)'
        )
    
    def _build_skill_patterns(self) -> List[str]:
        """Build common technical skill patterns."""
        return [
            # Programming languages
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
            "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
            # Frameworks
            "react", "angular", "vue", "django", "flask", "fastapi", "spring",
            "node.js", "express", ".net", "rails", "laravel", "nextjs", "nuxt",
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb", "oracle", "sqlite", "neo4j",
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
            "gitlab", "github actions", "ci/cd", "ansible", "prometheus", "grafana",
            # ML/AI
            "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
            "scikit-learn", "nlp", "computer vision", "llm", "transformers",
            # Other
            "git", "linux", "agile", "scrum", "rest api", "graphql", "microservices",
            "html", "css", "sass", "webpack", "nginx", "apache",
        ]
    
    async def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        try:
            with open_pdf(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        
        return "\n".join(text_parts)
    
    async def extract_text_from_docx(self, docx_bytes: bytes) -> str:
        """
        Extract text content from a DOCX file.
        
        Args:
            docx_bytes: DOCX file content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            doc = DocxDocument(io.BytesIO(docx_bytes))
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise
    
    async def extract_text(self, file_bytes: bytes, content_type: str) -> str:
        """
        Extract text from a document based on its content type.
        
        Args:
            file_bytes: File content as bytes
            content_type: MIME type of the file
            
        Returns:
            Extracted text content
        """
        if content_type == "application/pdf":
            return await self.extract_text_from_pdf(file_bytes)
        elif content_type in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            return await self.extract_text_from_docx(file_bytes)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    
    def extract_entities(self, text: str) -> List[ParsedEntity]:
        """
        Extract named entities using spaCy.
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted entities
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append(ParsedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
        
        return entities
    
    def extract_contact_info(self, text: str) -> ContactInfo:
        """
        Extract contact information from text.
        
        Args:
            text: Text to process
            
        Returns:
            Extracted contact information
        """
        contact = ContactInfo()
        
        # Extract email
        email_match = self._email_pattern.search(text)
        if email_match:
            contact.email = email_match.group()
        
        # Extract phone
        phone_match = self._phone_pattern.search(text)
        if phone_match:
            contact.phone = phone_match.group()
        
        # Extract LinkedIn
        linkedin_match = self._linkedin_pattern.search(text)
        if linkedin_match:
            contact.linkedin = linkedin_match.group()
        
        # Extract GitHub
        github_match = self._github_pattern.search(text)
        if github_match:
            contact.github = github_match.group()
        
        # Extract name using spaCy PERSON entities (first one is usually the candidate)
        doc = self.nlp(text[:1000])  # Only check first part of document
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                contact.name = ent.text
                break
        
        # Extract location using GPE entities
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                contact.location = ent.text
                break
        
        return contact
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching.
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted skills
        """
        text_lower = text.lower()
        found_skills = []
        
        for skill in self._skill_patterns:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Also extract from spaCy entities (PRODUCT, ORG can be technologies)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                skill_text = ent.text.lower().strip()
                if len(skill_text) > 1 and skill_text not in found_skills:
                    # Check if it might be a technology
                    if any(tech_word in skill_text for tech_word in ["js", "api", "db", "sql", "cloud"]):
                        found_skills.append(skill_text)
        
        return list(set(found_skills))
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract common resume sections.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary of section name to content
        """
        # Common section headers
        section_patterns = [
            (r'(?i)(?:professional\s+)?summary|objective|profile', 'summary'),
            (r'(?i)experience|work\s+history|employment', 'experience'),
            (r'(?i)education|academic', 'education'),
            (r'(?i)skills|technical\s+skills|competencies', 'skills'),
            (r'(?i)certifications?|licenses?', 'certifications'),
            (r'(?i)projects?', 'projects'),
            (r'(?i)languages?', 'languages'),
        ]
        
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            matched_section = None
            for pattern, section_name in section_patterns:
                if re.match(pattern, line_stripped) and len(line_stripped) < 50:
                    matched_section = section_name
                    break
            
            if matched_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = matched_section
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    async def parse_resume(self, file_bytes: bytes, content_type: str) -> ParsedResume:
        """
        Parse a resume document and extract structured data.
        
        Args:
            file_bytes: Resume file content
            content_type: MIME type of the file
            
        Returns:
            Parsed resume data
        """
        # Extract raw text
        raw_text = await self.extract_text(file_bytes, content_type)
        
        # Extract entities
        entities = self.extract_entities(raw_text)
        
        # Extract contact info
        contact = self.extract_contact_info(raw_text)
        
        # Extract skills
        skills = self.extract_skills(raw_text)
        
        # Extract sections
        sections = self.extract_sections(raw_text)
        
        # Build parsed resume
        parsed = ParsedResume(
            contact=contact,
            summary=sections.get('summary'),
            skills=skills,
            entities=entities,
            raw_text=raw_text,
        )
        
        # Extract certifications from section
        if 'certifications' in sections:
            cert_text = sections['certifications']
            # Split by newlines and filter empty
            certs = [c.strip() for c in cert_text.split('\n') if c.strip()]
            parsed.certifications = certs[:10]  # Limit to 10
        
        # Extract languages from section
        if 'languages' in sections:
            lang_text = sections['languages']
            langs = [l.strip() for l in re.split(r'[,\n]', lang_text) if l.strip()]
            parsed.languages = langs[:10]
        
        logger.info(f"Parsed resume: {len(skills)} skills, {len(entities)} entities")
        
        return parsed
    
    async def parse_job_description(self, text: str) -> ParsedJobDescription:
        """
        Parse a job description and extract structured data.
        
        Args:
            text: Job description text
            
        Returns:
            Parsed job description data
        """
        doc = self.nlp(text)
        
        # Extract basic info
        parsed = ParsedJobDescription(raw_text=text)
        
        # Try to find title (usually in first few lines)
        first_lines = text[:500]
        for ent in self.nlp(first_lines).ents:
            if not parsed.title and ent.label_ == "WORK_OF_ART":
                parsed.title = ent.text
            if not parsed.company and ent.label_ == "ORG":
                parsed.company = ent.text
            if not parsed.location and ent.label_ in ["GPE", "LOC"]:
                parsed.location = ent.text
        
        # Extract skills
        all_skills = self.extract_skills(text)
        
        # Categorize as required vs preferred
        text_lower = text.lower()
        required_section = self._find_section(text_lower, ['required', 'must have', 'requirements'])
        preferred_section = self._find_section(text_lower, ['preferred', 'nice to have', 'bonus'])
        
        if required_section:
            required_skills = self.extract_skills(required_section)
            parsed.required_skills = required_skills
            parsed.preferred_skills = [s for s in all_skills if s not in required_skills]
        else:
            parsed.required_skills = all_skills
        
        # Extract experience years
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', text_lower)
        if exp_match:
            parsed.experience_years_min = int(exp_match.group(1))
        
        # Extract responsibilities
        resp_section = self._find_section(text_lower, ['responsibilities', 'duties', 'what you will do'])
        if resp_section:
            responsibilities = self._extract_bullet_points(resp_section)
            parsed.responsibilities = responsibilities[:10]
        
        # Extract qualifications
        qual_section = self._find_section(text_lower, ['qualifications', 'requirements', 'what we need'])
        if qual_section:
            qualifications = self._extract_bullet_points(qual_section)
            parsed.qualifications = qualifications[:10]
        
        logger.info(f"Parsed JD: {len(parsed.required_skills)} required skills")
        
        return parsed
    
    def _find_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Find a section by keywords."""
        for keyword in keywords:
            pattern = rf'{keyword}[:\s]*\n(.+?)(?=\n\s*\n|\n[A-Z]|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        return None
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        # Match lines starting with bullets, dashes, or numbers
        bullet_pattern = r'(?:^|\n)\s*(?:•|[-–]|\d+[.)]\s*)\s*(.+?)(?=\n|$)'
        matches = re.findall(bullet_pattern, text)
        return [m.strip() for m in matches if m.strip()]


# Global processor instance (lazy loaded)
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
