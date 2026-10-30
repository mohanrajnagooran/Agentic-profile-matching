"""
skills.py — skill taxonomy, text normalisation and structured resume parsing.

Shared by the RAG layer, the requirement extractor and the ranking engine so that
"ReactJS", "React.js" and "react" are always treated as the same skill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date

# ---------------------------------------------------------------------------
# Skill taxonomy: canonical name -> (category, aliases)
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Frontend
    "React": ("frontend", ["react", "reactjs", "react.js"]),
    "Angular": ("frontend", ["angular", "angularjs"]),
    "Vue": ("frontend", ["vue", "vue.js", "vuejs"]),
    "Next.js": ("frontend", ["next.js", "nextjs"]),
    "Redux": ("frontend", ["redux"]),
    "HTML/CSS": ("frontend", ["html", "css", "html5", "css3", "html/css"]),
    "Tailwind": ("frontend", ["tailwind", "tailwindcss"]),
    "JavaScript": ("language", ["javascript", "js", "es6"]),
    "TypeScript": ("language", ["typescript", "ts"]),
    # Backend / languages
    "Node.js": ("backend", ["node.js", "nodejs", "node"]),
    "Express": ("backend", ["express", "express.js"]),
    "Python": ("language", ["python"]),
    "Django": ("backend", ["django"]),
    "Flask": ("backend", ["flask"]),
    "FastAPI": ("backend", ["fastapi"]),
    "Java": ("language", ["java"]),
    "Spring Boot": ("backend", ["spring boot", "spring"]),
    "Go": ("language", ["golang", "go"]),
    "C#": ("language", ["c#", "csharp"]),
    ".NET": ("backend", [".net", "dotnet", "asp.net"]),
    "Ruby on Rails": ("backend", ["rails", "ruby on rails"]),
    "Kotlin": ("language", ["kotlin"]),
    "Swift": ("language", ["swift"]),
    "GraphQL": ("backend", ["graphql"]),
    "REST APIs": ("backend", ["rest", "rest api", "rest apis", "restful"]),
    "Microservices": ("backend", ["microservices", "microservice"]),
    # Data
    "SQL": ("data", ["sql"]),
    "PostgreSQL": ("data", ["postgresql", "postgres"]),
    "MySQL": ("data", ["mysql"]),
    "MongoDB": ("data", ["mongodb", "mongo"]),
    "Redis": ("data", ["redis"]),
    "Kafka": ("data", ["kafka"]),
    "Spark": ("data", ["spark", "pyspark"]),
    "Airflow": ("data", ["airflow"]),
    "Pandas": ("data", ["pandas"]),
    "Tableau": ("data", ["tableau"]),
    # ML
    "Machine Learning": ("ml", ["machine learning", "ml"]),
    "Deep Learning": ("ml", ["deep learning"]),
    "TensorFlow": ("ml", ["tensorflow"]),
    "PyTorch": ("ml", ["pytorch"]),
    "scikit-learn": ("ml", ["scikit-learn", "sklearn"]),
    "NLP": ("ml", ["nlp", "natural language processing"]),
    "LLMs": ("ml", ["llm", "llms", "large language models", "langchain", "langgraph"]),
    # Cloud / DevOps
    "AWS": ("cloud", ["aws", "amazon web services"]),
    "Azure": ("cloud", ["azure"]),
    "GCP": ("cloud", ["gcp", "google cloud"]),
    "Docker": ("devops", ["docker", "containers"]),
    "Kubernetes": ("devops", ["kubernetes", "k8s"]),
    "Terraform": ("devops", ["terraform"]),
    "CI/CD": ("devops", ["ci/cd", "cicd", "github actions", "jenkins"]),
    "Linux": ("devops", ["linux"]),
    # Mobile
    "React Native": ("mobile", ["react native"]),
    "Flutter": ("mobile", ["flutter"]),
    "Android": ("mobile", ["android"]),
    "iOS": ("mobile", ["ios"]),
    # Testing / practice
    "Jest": ("testing", ["jest"]),
    "Cypress": ("testing", ["cypress"]),
    "Selenium": ("testing", ["selenium"]),
    "Unit Testing": ("testing", ["unit testing", "tdd"]),
    "Agile": ("practice", ["agile", "scrum"]),
    "System Design": ("practice", ["system design", "distributed systems"]),
    "Figma": ("design", ["figma"]),
}

# Aliases that are too ambiguous to match in free prose ("go to market", "node of").
# They still match inside a SKILLS line, where context makes them unambiguous.
_PROSE_UNSAFE = {"go", "js", "ts", "node", "ml", "rest", "spring", "ios", "sql", "containers"}

_ALIAS_TO_CANON: dict[str, str] = {}
for canon, (_cat, aliases) in SKILL_TAXONOMY.items():
    _ALIAS_TO_CANON[canon.lower()] = canon
    for a in aliases:
        _ALIAS_TO_CANON[a.lower()] = canon

# Longest aliases first so "react native" wins over "react", "spring boot" over "spring".
_SORTED_ALIASES = sorted(_ALIAS_TO_CANON, key=len, reverse=True)


def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(r"(?<![\w.+#/-])" + re.escape(alias) + r"(?![\w+#-]|\.\w)", re.IGNORECASE)


_ALIAS_PATTERNS = {a: _alias_pattern(a) for a in _SORTED_ALIASES}


def canonical_skill(name: str) -> str | None:
    """Map any alias ('reactjs', 'K8s') to its canonical skill name, or None."""
    key = name.strip().lower()
    if key in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[key]
    for alias in _SORTED_ALIASES:  # fuzzy: "react developer" -> React
        if _ALIAS_PATTERNS[alias].search(key) and alias not in _PROSE_UNSAFE:
            return _ALIAS_TO_CANON[alias]
    return None


def skill_category(skill: str) -> str:
    return SKILL_TAXONOMY.get(skill, ("other", []))[0]


def extract_skills(text: str, strict: bool = True) -> list[str]:
    """Find canonical skills mentioned in text, preserving first-seen order.

    strict=True skips ambiguous short aliases (use for prose);
    strict=False matches everything (use for comma-separated skill lists).
    """
    found: dict[str, int] = {}
    # "Kafka-based pipeline", "AWS-hosted" → treat the hyphenated suffix as a separate word
    masked = re.sub(r"-(based|powered|driven|backed|hosted|native|first|heavy|centric|enabled|focused|specific)\b", r" \1", text)
    for alias in _SORTED_ALIASES:
        if strict and alias in _PROSE_UNSAFE:
            continue
        pat = _ALIAS_PATTERNS[alias]
        for m in pat.finditer(masked):
            canon = _ALIAS_TO_CANON[alias]
            found.setdefault(canon, m.start())
        # mask matches so "react native" isn't re-counted as "react"
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)
    return [s for s, _ in sorted(found.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------
# Structured resume parsing
# ---------------------------------------------------------------------------
@dataclass
class Role:
    title: str
    company: str
    start_year: int
    end_year: int  # current year if "Present"
    current: bool
    bullets: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    months: int | None = None  # set when the resume gives month-level dates

    @property
    def years(self) -> float:
        if self.months is not None:
            return max(0.1, round(self.months / 12, 1))
        return max(0.5, self.end_year - self.start_year)


@dataclass
class Resume:
    candidate_id: str
    name: str
    email: str = ""
    title: str = ""
    location: str = ""
    total_years: float = 0.0
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    roles: list[Role] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_text", None)
        return d

    # --- derived evidence used by deep screening -------------------------
    def skill_years(self, skill: str) -> float:
        """Years of hands-on evidence for a skill, summed over roles that mention it."""
        return round(sum(r.years for r in self.roles if skill in r.skills), 1)

    def skill_recent(self, skill: str) -> bool:
        return any(r.current and skill in r.skills for r in self.roles)

    def all_skills(self) -> set[str]:
        s = set(self.skills)
        for r in self.roles:
            s.update(r.skills)
        return s

    def highest_degree(self) -> str:
        text = " ".join(self.education).lower()
        if re.search(r"\bph\.?d|doctor", text):
            return "PhD"
        if re.search(r"\bm\.?s\b|m\.?tech|master|mba|m\.?sc", text):
            return "Masters"
        if re.search(r"\bb\.?s\b|b\.?tech|b\.?e\b|bachelor|b\.?sc|b\.?a\b", text):
            return "Bachelors"
        if text.strip():
            return "Other"
        return "None listed"




# ---------------------------------------------------------------------------
# Resume parsing — works for the sample format ("Name: …", "Title, Company (2020 - Present)")
# and for free-form resumes extracted from PDF/DOCX (varied headings, month-level dates,
# "Title | Company | Jan 2020 – Present", dates on their own line, missing skills section, …).
# ---------------------------------------------------------------------------
_SECTION_NAMES = {
    "SUMMARY": ["summary", "profile", "professional summary", "profile summary", "career summary", "about me", "about",
                "objective", "career objective", "professional profile", "overview"],
    "SKILLS": ["skills", "technical skills", "key skills", "core skills", "core competencies", "competencies", "skill set",
               "skillset", "tech stack", "technologies", "technical expertise", "skills & tools", "skills and tools",
               "tools & technologies", "tools and technologies", "technical proficiency", "areas of expertise", "expertise"],
    "EXPERIENCE": ["experience", "work experience", "professional experience", "employment", "employment history",
                   "work history", "career history", "relevant experience", "professional background", "internships",
                   "experience & internships", "internship experience", "industry experience"],
    "EDUCATION": ["education", "academic background", "academics", "education & training", "education and training",
                  "academic qualifications", "educational qualifications", "qualifications"],
    "CERTIFICATIONS": ["certifications", "certification", "certificates", "licenses & certifications",
                       "licenses and certifications", "courses", "courses & certifications", "certifications & courses"],
    "PROJECTS": ["projects", "personal projects", "key projects", "academic projects", "selected projects", "side projects"],
    "OTHER": ["achievements", "awards", "honors", "honours", "interests", "hobbies", "languages", "publications",
              "references", "volunteer", "volunteering", "extracurricular activities", "activities", "declaration",
              "personal details", "personal information", "contact", "strengths", "leadership"],
}
_HEADER_LOOKUP = {name: key for key, names in _SECTION_NAMES.items() for name in names}

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_POINT = rf"(?:{_MON}\s*['’]?\s*\d{{4}}|\d{{1,2}}\s*[/.-]\s*\d{{4}}|\d{{4}}\s*[/.-]\s*\d{{1,2}}(?!\d)|(?:19|20)\d{{2}})"
_DATE_RANGE = re.compile(
    rf"\(?\s*(?P<start>{_POINT})\s*(?:-|–|—|to|till|until|→)+\s*(?P<end>{_POINT}|present|current|now|today|ongoing|date)\s*\)?",
    re.IGNORECASE,
)
_ROLE_WORDS = re.compile(r"\b(engineer|developer|intern|manager|lead|architect|analyst|scientist|consultant|designer|sde|"
                         r"programmer|administrator|specialist|head|director|tester|qa|devops|sre|associate|trainee|"
                         r"officer|founder|cto|vp|executive|coordinator)\b", re.I)
_BULLET_CHARS = "-•*●▪◦■□➢✓►‣⁃–·o"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_YEARS_EXP = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b(?:\s+of)?(?:\s+\w+){0,3}?\s+(?:experience|exp)\b", re.I)


def _header_field(text: str, label: str) -> str:
    m = re.search(rf"^\s*{label}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _section_key(line: str) -> str | None:
    """Return the canonical section for a heading line, e.g. 'WORK EXPERIENCE:' -> 'EXPERIENCE'."""
    t = re.sub(r"[\s:|_=~*#-]+$", "", line.strip()).strip(" #*_=-:|")
    t = re.sub(r"\s+", " ", t).lower()
    if not t or len(t) > 45:
        return None
    if t in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[t]
    spaced = t.replace(" ", "")  # PDFs sometimes letter-space headings: "E X P E R I E N C E"
    for name, key in _HEADER_LOOKUP.items():
        if spaced == name.replace(" ", ""):
            return key
    return None


def _to_month_index(point: str, end: bool = False) -> int | None:
    """Convert a date point to months since year 0. Year-only starts at Jan, year-only ends at Dec."""
    p = point.strip().lower()
    if p in ("present", "current", "now", "today", "ongoing", "date"):
        t = date.today()
        return t.year * 12 + t.month
    m = re.match(rf"({_MON})\s*['’]?\s*(\d{{4}})", p)
    if m:
        return int(m.group(2)) * 12 + _MONTHS[m.group(1)[:3]]
    m = re.match(r"(\d{1,2})\s*[/.-]\s*(\d{4})", p)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)) * 12 + int(m.group(1))
    m = re.match(r"(\d{4})\s*[/.-]\s*(\d{1,2})", p)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"(\d{4})", p)
    if m:
        return int(m.group(1)) * 12 + (12 if end else 1)
    return None


def _has_month(point: str) -> bool:
    return not re.fullmatch(r"\s*\d{4}\s*", point)


def _clean_bullet(line: str) -> str:
    return line.strip().lstrip(_BULLET_CHARS).strip() if re.match(rf"^\s*[{re.escape(_BULLET_CHARS)}]\s", line + " ") or \
        line.strip()[:1] in "•●▪◦■➢✓►‣" else line.strip()


def _split_title_company(text: str) -> tuple[str, str]:
    text = re.sub(r"[()\[\]]", " ", text)
    text = re.sub(r"\s{2,}", " | ", text.strip())  # wide gaps from PDF column layout act as separators
    text = text.strip(" ,|-–—·@\t")
    parts = [p.strip(" ,|-–—·\t") for p in re.split(r"\s+(?:at|@)\s+|\s*[|·•]\s*|\s+[-–—]\s+|,\s*", text) if p.strip(" ,|-–—·\t")]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _looks_like_name(line: str) -> bool:
    s = line.strip()
    if not s or _EMAIL.search(s) or re.search(r"\d{3}|https?:|www\.|linkedin|github", s, re.I) or _section_key(s):
        return False
    words = s.replace(".", " ").split()
    return 1 <= len(words) <= 5 and all(re.fullmatch(r"[A-Za-z'’-]+", w) for w in words)


def _guess_name(lines: list[str], fallback: str) -> str:
    for line in lines[:8]:
        if _looks_like_name(line):
            name = re.sub(r"\s+", " ", line.strip())
            return name.title() if name.isupper() else name
        first = re.split(r"\s{2,}|\s[|·•]\s", line.strip())[0]  # "Jane Doe | jane@x.com | +91…"
        if first != line.strip() and _looks_like_name(first):
            return first.title() if first.isupper() else first
    return fallback


def _merge_months(intervals: list[tuple[int, int]]) -> int:
    total, cur = 0, None
    for s, e in sorted(intervals):
        if cur and s <= cur[1]:
            cur = (cur[0], max(cur[1], e))
        else:
            if cur:
                total += cur[1] - cur[0]
            cur = (s, e)
    if cur:
        total += cur[1] - cur[0]
    return total


def parse_resume(text: str, candidate_id: str, source_path: str = "") -> Resume:
    """Parse resume text into a Resume. Tolerant of missing sections and many real-world layouts."""
    text = text.replace("\r", "\n").replace(" ", " ").replace(" ", " ").replace(" ", " ")
    lines = [l.rstrip() for l in text.splitlines()]
    today = date.today()

    # ---- split into sections (lines before the first heading form the header block)
    sections: dict[str, list[str]] = {}
    header_block: list[str] = []
    key = None
    for line in lines:
        k = _section_key(line) if line.strip() else None
        if k:
            key = k
            sections.setdefault(key, [])
            continue
        (sections[key] if key else header_block).append(line)
    sec = {k: "\n".join(v).strip() for k, v in sections.items()}

    name = _header_field(text, "Name") or _guess_name([l for l in lines if l.strip()], candidate_id)
    email = _header_field(text, "Email") or (_EMAIL.search(text).group(0) if _EMAIL.search(text) else "")

    # ---- skills: the skills section if present, else anything recognisable in the whole resume
    if sec.get("SKILLS"):
        skills = extract_skills(sec["SKILLS"], strict=False)
    else:
        skills = extract_skills(text, strict=True)

    # ---- work history
    roles: list[Role] = []
    current: Role | None = None
    intervals: list[tuple[int, int]] = []  # month spans, for total experience without double-counting overlaps
    pending: list[str] = []  # short non-bullet lines seen since the last role/bullet (title lines above dates)
    exp_lines = sections.get("EXPERIENCE", [])
    for raw in exp_lines:
        line = raw.strip()
        if not line:
            continue
        dm = _DATE_RANGE.search(line)
        is_bullet = line[:1] in _BULLET_CHARS and (len(line) > 1 and (line[1] == " " or line[0] in "•●▪◦■➢✓►‣"))
        if dm and not is_bullet and len(line.split()) <= 18:
            s_idx = _to_month_index(dm.group("start"))
            e_idx = _to_month_index(dm.group("end"), end=True)
            if s_idx is None or e_idx is None or e_idx < s_idx:
                continue
            rest = (line[:dm.start()] + "  " + line[dm.end():]).strip()
            title, company = _split_title_company(rest)
            if pending:  # "Software Engineer" on the line above "Acme Corp, City   Jan 2020 – Present"
                above_title, above_company = _split_title_company(pending[-1])
                if not title:
                    title, company = above_title, above_company
                elif _ROLE_WORDS.search(above_title) and not _ROLE_WORDS.search(title):
                    title, company = above_title, title
            month_level = _has_month(dm.group("start")) or _has_month(dm.group("end"))
            end_raw = dm.group("end").strip().lower()
            is_current = end_raw in ("present", "current", "now", "today", "ongoing", "date")
            # year-only ranges keep "end - start" semantics (2019 - 2022 = 3 years); month ranges are exact
            current = Role(title=title or "Role", company=company,
                           start_year=(s_idx - 1) // 12,
                           end_year=today.year if is_current else (e_idx - 1) // 12,
                           current=is_current, months=(e_idx - s_idx + 1) if month_level else None)
            current.skills = extract_skills(f"{title} {company}")
            roles.append(current)
            intervals.append((s_idx - 1, e_idx) if month_level else (current.start_year * 12, current.end_year * 12))
            pending = []
            continue
        cleaned = _clean_bullet(line)
        if current and (is_bullet or len(cleaned.split()) > 6):
            current.bullets.append(cleaned)
            for s in extract_skills(cleaned):
                if s not in current.skills:
                    current.skills.append(s)
            pending = []
        elif current and current.bullets and cleaned[:1].islower():  # wrapped continuation of the previous bullet
            current.bullets[-1] += " " + cleaned
            for s in extract_skills(cleaned):
                if s not in current.skills:
                    current.skills.append(s)
        else:
            pending.append(cleaned)

    # ---- total experience
    years_field = _header_field(text, "Total Experience")
    ym = re.search(r"(\d+(?:\.\d+)?)", years_field)
    stated = _YEARS_EXP.search(sec.get("SUMMARY", "") + "\n" + "\n".join(header_block))
    if ym:
        total = float(ym.group(1))
    elif roles:
        total = round(_merge_months(intervals) / 12, 1)
        if stated and abs(float(stated.group(1)) - total) > 1 and total < 0.5:
            total = float(stated.group(1))
    elif stated:
        total = float(stated.group(1))
    else:
        total = 0.0

    edu = [_clean_bullet(l) for l in sections.get("EDUCATION", []) if l.strip() and _clean_bullet(l) not in ("", "-")]
    certs = [_clean_bullet(l) for l in sections.get("CERTIFICATIONS", []) if l.strip()]
    title = _header_field(text, "Title") or next((r.title for r in roles if r.current), roles[0].title if roles else "")
    summary = sec.get("SUMMARY", "")
    if not summary:  # free-form resumes often put a one-paragraph intro in the header block
        long_lines = [l.strip() for l in header_block if len(l.split()) >= 8 and not _EMAIL.search(l)]
        summary = " ".join(long_lines)

    return Resume(
        candidate_id=candidate_id,
        name=name,
        email=email,
        title=title,
        location=_header_field(text, "Location"),
        total_years=total,
        summary=" ".join(summary.split()),
        skills=skills,
        roles=roles,
        education=edu,
        certifications=certs,
        raw_text=text,
        source_path=source_path,
    )
