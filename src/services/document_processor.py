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
    
    # =========================================================================
    # LLM-Based Parsing Methods
    # =========================================================================
    
    async def parse_resume_with_llm(
        self,
        file_bytes: bytes,
        content_type: str,
        use_cache: bool = True,
    ) -> ParsedResume:
        """
        Parse a resume using LLM for robust extraction.
        
        This method uses an LLM to extract structured information from resume text,
        handling diverse formats and layouts that rule-based parsing may miss.
        
        Args:
            file_bytes: File content as bytes
            content_type: MIME type of the file
            use_cache: Whether to use cached results if available
            
        Returns:
            ParsedResume with extracted information
        """
        import hashlib
        import json
        
        # Generate cache key from file hash
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        cache_key = f"resume_{file_hash}"
        
        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Using cached resume parse: {cache_key}")
                return cached
        
        # Extract raw text first
        text = await self.extract_text(file_bytes, content_type)
        
        if not text or len(text.strip()) < 50:
            logger.warning("Resume text too short, returning empty parse")
            return ParsedResume(raw_text=text)
        
        try:
            from src.providers.llm import get_llm_provider
            from src.providers.llm.base import GenerationConfig, user_message, system_message
            from src.services.prompts import RESUME_EXTRACTION_PROMPT
            
            llm = await get_llm_provider()
            
            # Truncate text if too long (keep first 8000 chars for context window)
            resume_text = text[:8000] if len(text) > 8000 else text
            
            prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
            
            messages = [
                system_message("You are an expert resume parser. Extract information accurately and completely."),
                user_message(prompt),
            ]
            
            logger.info("Parsing resume with LLM...")
            response = await llm.generate(
                messages,
                GenerationConfig(max_tokens=2048, temperature=0.1)
            )
            
            # Parse JSON response
            data = self._parse_llm_json_response(response.content)
            
            if not data:
                logger.warning("LLM returned invalid JSON, falling back to rule-based parsing")
                return await self.parse_resume(file_bytes, content_type)
            
            # Convert to ParsedResume
            parsed = self._json_to_parsed_resume(data, text)
            
            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, parsed)
            
            logger.info(f"LLM resume parse complete: {parsed.contact.name if parsed.contact else 'Unknown'}")
            return parsed
            
        except Exception as e:
            logger.warning(f"LLM resume parsing failed: {e}, falling back to rule-based")
            return await self.parse_resume(file_bytes, content_type)
    
    async def parse_job_description_with_llm(
        self,
        text: str,
        use_cache: bool = True,
    ) -> ParsedJobDescription:
        """
        Parse a job description using LLM for robust extraction.
        
        Args:
            text: Job description text
            use_cache: Whether to use cached results if available
            
        Returns:
            ParsedJobDescription with extracted information
        """
        import hashlib
        import json
        
        # Generate cache key from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"jd_{text_hash}"
        
        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Using cached JD parse: {cache_key}")
                return cached
        
        if not text or len(text.strip()) < 50:
            logger.warning("JD text too short, returning empty parse")
            return ParsedJobDescription(raw_text=text)
        
        try:
            from src.providers.llm import get_llm_provider
            from src.providers.llm.base import GenerationConfig, user_message, system_message
            from src.services.prompts import JD_EXTRACTION_PROMPT
            
            llm = await get_llm_provider()
            
            # Truncate if too long
            jd_text = text[:8000] if len(text) > 8000 else text
            
            prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text)
            
            messages = [
                system_message("You are an expert job description parser. Extract requirements accurately."),
                user_message(prompt),
            ]
            
            logger.info("Parsing job description with LLM...")
            response = await llm.generate(
                messages,
                GenerationConfig(max_tokens=2048, temperature=0.1)
            )
            
            # Parse JSON response
            data = self._parse_llm_json_response(response.content)
            
            if not data:
                logger.warning("LLM returned invalid JSON, falling back to rule-based parsing")
                return await self.parse_job_description(text)
            
            # Convert to ParsedJobDescription
            parsed = self._json_to_parsed_jd(data, text)
            
            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, parsed)
            
            logger.info(f"LLM JD parse complete: {parsed.title or 'Unknown'}")
            return parsed
            
        except Exception as e:
            logger.warning(f"LLM JD parsing failed: {e}, falling back to rule-based")
            return await self.parse_job_description(text)
    
    def _parse_llm_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common formatting issues."""
        import json
        
        content = content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        # Find JSON object bounds
        if not content.startswith("{"):
            start = content.find("{")
            if start != -1:
                content = content[start:]
        
        if not content.endswith("}"):
            end = content.rfind("}")
            if end != -1:
                content = content[:end + 1]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            logger.debug(f"Content was: {content[:500]}")
            return None
    
    def _json_to_parsed_resume(self, data: Dict[str, Any], raw_text: str) -> ParsedResume:
        """Convert LLM JSON output to ParsedResume model."""
        # Extract contact info
        contact_data = data.get("contact", {})
        contact = ContactInfo(
            name=contact_data.get("full_name"),
            email=contact_data.get("email"),
            phone=contact_data.get("phone"),
            linkedin=contact_data.get("linkedin"),
            github=contact_data.get("github"),
            location=contact_data.get("location"),
        )
        
        # Extract experience
        experience = []
        for exp_data in data.get("experience", []):
            if isinstance(exp_data, dict):
                experience.append(Experience(
                    company=exp_data.get("company"),
                    title=exp_data.get("title"),
                    location=exp_data.get("location"),
                    start_date=exp_data.get("start_date"),
                    end_date=exp_data.get("end_date"),
                    description=exp_data.get("description"),
                    highlights=exp_data.get("highlights", []),
                ))
        
        # Extract education
        education = []
        for edu_data in data.get("education", []):
            if isinstance(edu_data, dict):
                education.append(Education(
                    institution=edu_data.get("institution"),
                    degree=edu_data.get("degree"),
                    field=edu_data.get("field"),
                    start_date=edu_data.get("start_date"),
                    end_date=edu_data.get("end_date"),
                    gpa=edu_data.get("gpa"),
                ))
        
        # Extract skills
        skills = data.get("skills", [])
        if isinstance(skills, list):
            skills = [s for s in skills if isinstance(s, str)]
        else:
            skills = []
        
        # Extract certifications
        certifications = data.get("certifications", [])
        if not isinstance(certifications, list):
            certifications = []
        
        return ParsedResume(
            contact=contact,
            summary=data.get("summary"),
            skills=skills,
            experience=experience,
            education=education,
            certifications=certifications,
            raw_text=raw_text,
        )
    
    def _json_to_parsed_jd(self, data: Dict[str, Any], raw_text: str) -> ParsedJobDescription:
        """Convert LLM JSON output to ParsedJobDescription model."""
        return ParsedJobDescription(
            title=data.get("title"),
            company=data.get("company"),
            location=data.get("location"),
            employment_type=data.get("employment_type"),
            experience_level=data.get("experience_level"),
            experience_years_min=data.get("experience_years_min"),
            experience_years_max=data.get("experience_years_max"),
            salary_min=data.get("salary_min"),
            salary_max=data.get("salary_max"),
            salary_currency=data.get("salary_currency"),
            required_skills=data.get("required_skills", []),
            preferred_skills=data.get("preferred_skills", []),
            responsibilities=data.get("responsibilities", []),
            qualifications=data.get("qualifications", []),
            raw_text=raw_text,
        )
    
    # =========================================================================
    # Caching Methods
    # =========================================================================
    
    _cache_dir: Optional[Path] = None
    
    @classmethod
    def _get_cache_dir(cls) -> Path:
        """Get or create the cache directory."""
        if cls._cache_dir is None:
            cache_path = Path(__file__).parent.parent.parent / ".cache" / "parsed_documents"
            cache_path.mkdir(parents=True, exist_ok=True)
            cls._cache_dir = cache_path
        return cls._cache_dir
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get a cached parsed document."""
        import json
        
        cache_file = self._get_cache_dir() / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Reconstruct the appropriate model
            if cache_key.startswith("resume_"):
                return self._json_to_parsed_resume(data, data.get("_raw_text", ""))
            elif cache_key.startswith("jd_"):
                return self._json_to_parsed_jd(data, data.get("_raw_text", ""))
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, parsed: Any) -> None:
        """Save a parsed document to cache."""
        import json
        
        cache_file = self._get_cache_dir() / f"{cache_key}.json"
        
        try:
            # Convert model to dict for caching
            if isinstance(parsed, ParsedResume):
                data = {
                    "contact": {
                        "full_name": parsed.contact.name if parsed.contact else None,
                        "email": parsed.contact.email if parsed.contact else None,
                        "phone": parsed.contact.phone if parsed.contact else None,
                        "linkedin": parsed.contact.linkedin if parsed.contact else None,
                        "github": parsed.contact.github if parsed.contact else None,
                        "location": parsed.contact.location if parsed.contact else None,
                    },
                    "summary": parsed.summary,
                    "skills": parsed.skills,
                    "experience": [
                        {
                            "company": exp.company,
                            "title": exp.title,
                            "location": exp.location,
                            "start_date": exp.start_date,
                            "end_date": exp.end_date,
                            "description": exp.description,
                            "highlights": exp.highlights,
                        }
                        for exp in parsed.experience
                    ],
                    "education": [
                        {
                            "institution": edu.institution,
                            "degree": edu.degree,
                            "field": edu.field,
                            "start_date": edu.start_date,
                            "end_date": edu.end_date,
                            "gpa": edu.gpa,
                        }
                        for edu in parsed.education
                    ],
                    "certifications": parsed.certifications,
                    "_raw_text": parsed.raw_text,
                }
            elif isinstance(parsed, ParsedJobDescription):
                data = {
                    "title": parsed.title,
                    "company": parsed.company,
                    "location": parsed.location,
                    "employment_type": parsed.employment_type,
                    "experience_level": parsed.experience_level,
                    "experience_years_min": parsed.experience_years_min,
                    "experience_years_max": parsed.experience_years_max,
                    "salary_min": parsed.salary_min,
                    "salary_max": parsed.salary_max,
                    "salary_currency": parsed.salary_currency,
                    "required_skills": parsed.required_skills,
                    "preferred_skills": parsed.preferred_skills,
                    "responsibilities": parsed.responsibilities,
                    "qualifications": parsed.qualifications,
                    "_raw_text": parsed.raw_text,
                }
            else:
                return
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved to cache: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def clear_cache(self) -> int:
        """Clear all cached parsed documents. Returns number of files deleted."""
        cache_dir = self._get_cache_dir()
        count = 0
        for cache_file in cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        logger.info(f"Cleared {count} cached documents")
        return count


# Global processor instance (lazy loaded)
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
