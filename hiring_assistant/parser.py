"""
parser.py — Advanced resume parser with multi-strategy extraction,
            confidence scoring, and structured work history detection.
"""

from __future__ import annotations
import re
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from .models import Candidate, WorkExperience, EducationLevel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  SKILL TAXONOMY  (grouped by domain for weighted matching)
# ─────────────────────────────────────────────────────────────

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "ruby", "go", "golang", "rust", "swift", "kotlin", "scala", "r",
        "php", "perl", "matlab", "dart", "elixir", "haskell",
    ],
    "web_frontend": [
        "react", "angular", "vue", "svelte", "next.js", "nuxt.js",
        "html", "css", "sass", "tailwind", "bootstrap", "webpack",
        "redux", "graphql", "rest api", "ui/ux", "figma",
    ],
    "web_backend": [
        "django", "flask", "fastapi", "spring", "node.js", "express",
        "rails", "laravel", "asp.net", "fastify", "gin", "echo",
        "microservices", "grpc", "websockets",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "sqlite", "oracle", "mssql",
        "mongodb", "redis", "cassandra", "dynamodb", "elasticsearch",
        "neo4j", "influxdb", "firebase",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "github actions", "ci/cd", "linux",
        "bash", "nginx", "apache", "prometheus", "grafana", "helm",
    ],
    "data_ml": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "scikit-learn", "keras", "pandas",
        "numpy", "matplotlib", "seaborn", "spark", "hadoop",
        "data analysis", "power bi", "tableau", "excel", "airflow",
        "mlflow", "hugging face",
    ],
    "practices": [
        "agile", "scrum", "kanban", "tdd", "bdd", "devops", "git",
        "jira", "confluence", "code review", "system design",
        "object-oriented", "functional programming", "design patterns",
        "api design", "security", "testing", "ci/cd",
    ],
    "soft_skills": [
        "leadership", "communication", "problem solving", "teamwork",
        "mentoring", "project management", "stakeholder management",
    ],
}

# Flattened set for fast lookup
ALL_SKILLS: List[str] = [s for group in SKILL_TAXONOMY.values() for s in group]

# Section header patterns
SECTION_PATTERNS = {
    "summary": re.compile(
        r"^(?:summary|profile|about|objective|overview|professional\s+summary)[:\s]*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "skills": re.compile(
        r"^(?:skills?|technical\s+skills?|core\s+competencies|technologies)[:\s]*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "experience": re.compile(
        r"^(?:experience|work\s+experience|employment|work\s+history|professional\s+experience)[:\s]*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "education": re.compile(
        r"^(?:education|academic|qualifications)[:\s]*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "certifications": re.compile(
        r"^(?:certifications?|certificates?|licenses?|courses?)[:\s]*$",
        re.IGNORECASE | re.MULTILINE
    ),
}


class ResumeParser:
    """
    Advanced resume parser using multi-strategy NLP-style extraction.

    Strategies used (in priority order):
      1. Section-aware parsing — locate named sections first
      2. Pattern matching — regex for emails, phones, dates, URLs
      3. Heuristic scoring — estimate parse confidence
    """

    def __init__(self, custom_skills: Optional[List[str]] = None):
        """
        Args:
            custom_skills: Extra skills to add to the taxonomy.
        """
        self._skills = ALL_SKILLS.copy()
        if custom_skills:
            self._skills.extend([s.lower().strip() for s in custom_skills])

    # ──────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────

    def parse(self, text: str) -> Candidate:
        """
        Parse raw resume text into a structured Candidate.

        Args:
            text: Raw resume string (plain text).

        Returns:
            Candidate: Populated profile.

        Raises:
            ValueError: If text is empty or too short to parse.
        """
        if not text or len(text.strip()) < 20:
            raise ValueError("Resume text is too short or empty to parse.")

        text = self._normalize(text)
        sections = self._split_sections(text)

        candidate = Candidate()
        candidate.raw_text = text

        candidate.name         = self._extract_name(text)
        candidate.email        = self._extract_email(text)
        candidate.phone        = self._extract_phone(text)
        candidate.linkedin     = self._extract_linkedin(text)
        candidate.github       = self._extract_github(text)
        candidate.location     = self._extract_location(text)
        candidate.skills       = self._extract_skills(text, sections)
        candidate.work_history = self._extract_work_history(
            sections.get("experience", text)
        )
        candidate.experience_years = self._calculate_total_experience(
            candidate.work_history, text
        )
        candidate.education        = self._extract_education_level(text)
        candidate.education_field  = self._extract_education_field(text)
        candidate.certifications   = self._extract_certifications(text, sections)
        candidate.languages        = self._extract_languages(text)
        candidate.summary          = self._extract_summary(text, sections)
        candidate.parse_confidence = self._calculate_confidence(candidate)

        logger.info(
            "Parsed candidate '%s' — confidence %.0f%%",
            candidate.name, candidate.parse_confidence * 100
        )
        return candidate

    def parse_from_file(self, filepath: str) -> Candidate:
        """Parse a resume from a .txt file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                return self.parse(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Resume file not found: {filepath}")

    # ──────────────────────────────────────────────────
    #  PRIVATE HELPERS
    # ──────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Clean up whitespace and encoding artifacts."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\t", "  ", text)
        text = re.sub(r" {3,}", "  ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    def _split_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and split resume into named sections.
        Returns dict: section_name → section_content.
        """
        sections: Dict[str, str] = {}
        lines = text.splitlines()
        current_section = "header"
        current_lines: List[str] = []

        for line in lines:
            matched_section = None
            for name, pattern in SECTION_PATTERNS.items():
                if pattern.match(line.strip()):
                    matched_section = name
                    break

            if matched_section:
                sections[current_section] = "\n".join(current_lines)
                current_section = matched_section
                current_lines = []
            else:
                current_lines.append(line)

        sections[current_section] = "\n".join(current_lines)
        return sections

    def _extract_name(self, text: str) -> str:
        """Extract candidate name from the header region."""
        lines = [l.strip() for l in text.splitlines()[:8] if l.strip()]
        for line in lines:
            # Skip lines with contact info markers
            if re.search(r"[@|•\d()\[\]/\\]", line):
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                # Filter common header labels
                if not re.match(r"(?:resume|curriculum|vitae|cv|profile)", line, re.IGNORECASE):
                    return line
        return "Unknown"

    def _extract_email(self, text: str) -> str:
        m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        return m.group(0).lower() if m else ""

    def _extract_phone(self, text: str) -> str:
        patterns = [
            r"\+\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{4}",            # +1-555-0101
            r"\+?1?[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}",  # (555) 010-1234
            r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",                  # 555-010-1234
            r"\b\d{10}\b",                                        # 5550101234
        ]
        for p in patterns:
            for m in re.finditer(p, text):
                digits = re.sub(r"\D", "", m.group(0))
                if 7 <= len(digits) <= 15:
                    return re.sub(r"\s+", " ", m.group(0).strip())
        return ""

    def _extract_linkedin(self, text: str) -> str:
        m = re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
        return f"https://www.{m.group(0)}" if m else ""

    def _extract_github(self, text: str) -> str:
        m = re.search(r"github\.com/[\w\-]+", text, re.IGNORECASE)
        return f"https://www.{m.group(0)}" if m else ""

    def _extract_location(self, text: str) -> str:
        """Extract city/country from header lines."""
        lines = [l.strip() for l in text.splitlines()[:10] if l.strip()]
        for line in lines:
            # Looks like "New York, NY" or "Bangalore, India"
            if re.match(r"^[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+$", line):
                return line
        return ""

    def _extract_skills(self, text: str, sections: Dict[str, str]) -> List[str]:
        """
        Extract skills using both keyword matching and section-aware parsing.
        Section-specific text is searched first to boost confidence.
        """
        text_lower = text.lower()
        section_text = sections.get("skills", "").lower()
        found = set()

        for skill in self._skills:
            pattern = r"\b" + re.escape(skill) + r"\b"
            # Extra weight to explicit skills sections, but search full text too
            if re.search(pattern, text_lower):
                found.add(skill)

        # Also capture comma/bullet-separated skills from the skills section
        if section_text:
            tokens = re.split(r"[,•\|\n/]+", section_text)
            for token in tokens:
                token = token.strip()
                if 2 <= len(token) <= 30:
                    for skill in self._skills:
                        if skill in token:
                            found.add(skill)

        return sorted(found)

    def _extract_work_history(self, experience_text: str) -> List[WorkExperience]:
        """Parse individual work entries from experience section."""
        entries: List[WorkExperience] = []
        current_year = datetime.now().year

        # Split into blocks by date patterns
        blocks = re.split(
            r"\n(?=[A-Z][^\n]{5,60}\n|[A-Z][^\n]{5,60}\s+\d{4})",
            experience_text
        )

        for block in blocks:
            if not block.strip():
                continue

            # Extract year range
            year_match = re.search(
                r"(\d{4})\s*[-–—]\s*(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)",
                block
            )
            if not year_match:
                continue

            start_year = int(year_match.group(1))
            end_raw = year_match.group(2).lower()
            end_year = None if end_raw in ("present", "current", "now") else int(end_raw)

            # Extract company name (usually first capitalised line)
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            company = ""
            title = ""
            for line in lines[:3]:
                line_clean = re.sub(r"\d{4}.*", "", line).strip()
                if not line_clean:
                    continue
                if not company:
                    company = line_clean
                elif not title:
                    title = line_clean
                else:
                    break

            description = " ".join(
                l.strip() for l in lines[3:8] if l.strip()
            )

            if start_year >= 1970:
                entries.append(WorkExperience(
                    company=company,
                    title=title,
                    start_year=start_year,
                    end_year=end_year,
                    description=description[:300],
                ))

        # Sort most recent first
        entries.sort(key=lambda e: e.start_year, reverse=True)
        return entries

    def _calculate_total_experience(
        self, work_history: List[WorkExperience], text: str
    ) -> float:
        """
        Calculate total years of experience.
        Uses work history timeline first, falls back to explicit mentions.
        """
        if work_history:
            # Sum non-overlapping durations (simplified: take the span of career)
            years = [w.start_year for w in work_history]
            if years:
                earliest = min(years)
                return round(min(datetime.now().year - earliest, 40), 1)

        # Fallback: explicit mention
        direct = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", text, re.IGNORECASE)
        if direct:
            return float(max(int(y) for y in direct))

        return 0.0

    def _extract_education_level(self, text: str) -> EducationLevel:
        """Return highest detected education level."""
        return EducationLevel.from_string(text)

    def _extract_education_field(self, text: str) -> str:
        """Extract field of study (e.g., 'Computer Science')."""
        patterns = [
            r"(?:in|of)\s+([A-Z][a-zA-Z\s]{3,40}?)(?:,|\.|from|\d)",
            r"B\.?(?:Sc|Tech|E)\.?\s+(?:in\s+)?([A-Z][a-zA-Z\s]{3,30})",
            r"M\.?(?:Sc|Tech|BA|S)\.?\s+(?:in\s+)?([A-Z][a-zA-Z\s]{3,30})",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_certifications(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Extract certification names."""
        cert_text = sections.get("certifications", "")
        # Also look in the full text for known cert patterns
        combined = cert_text + "\n" + text
        patterns = [
            r"AWS\s+(?:Certified\s+)?[A-Z][a-zA-Z\s]{3,40}",
            r"Google\s+(?:Cloud\s+)?(?:Certified\s+)?[A-Z][a-zA-Z\s]{3,30}",
            r"Microsoft\s+(?:Certified\s+)?[A-Z][a-zA-Z:\s]{3,40}",
            r"(?:PMP|CISSP|CPA|CISA|CEH|OSCP|CCNA|CCNP|CKA|GCP|Azure)\b",
            r"(?:Certified\s+[A-Z][a-zA-Z\s]{3,30})",
        ]
        found = set()
        for p in patterns:
            for m in re.finditer(p, combined):
                cert = m.group(0).strip()
                if len(cert) > 3:
                    found.add(cert)
        return sorted(found)[:10]

    def _extract_languages(self, text: str) -> List[str]:
        """Extract spoken/written languages (not programming languages)."""
        human_langs = [
            "english", "spanish", "french", "german", "mandarin", "chinese",
            "japanese", "arabic", "hindi", "portuguese", "russian", "korean",
            "italian", "dutch", "tamil", "telugu", "kannada",
        ]
        text_lower = text.lower()
        found = []
        # Only look in sections that mention languages explicitly
        lang_section = re.search(
            r"(?:languages?|linguistic)[:\s]+(.{0,200})", text, re.IGNORECASE
        )
        search_text = lang_section.group(1).lower() if lang_section else text_lower
        for lang in human_langs:
            if re.search(r"\b" + lang + r"\b", search_text):
                found.append(lang.title())
        return found

    def _extract_summary(self, text: str, sections: Dict[str, str]) -> str:
        """Extract the professional summary/objective."""
        # Try dedicated section first
        if "summary" in sections and sections["summary"].strip():
            raw = sections["summary"].strip()
            return " ".join(raw.split())[:400]

        # Fallback: first substantial paragraph
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if len(p.strip()) > 80]
        if paragraphs:
            return " ".join(paragraphs[0].split())[:400]
        return ""

    def _calculate_confidence(self, c: Candidate) -> float:
        """
        Heuristic parse confidence score (0–1).
        Checks how many key fields were successfully extracted.
        """
        score = 0.0
        checks = [
            (c.name != "Unknown", 0.20),
            (bool(c.email), 0.20),
            (bool(c.skills), 0.20),
            (c.experience_years > 0, 0.15),
            (c.education != EducationLevel.UNKNOWN, 0.15),
            (bool(c.summary), 0.10),
        ]
        for condition, weight in checks:
            if condition:
                score += weight
        return round(score, 2)
