"""
fs_tools.py — Milestone 1 file-system tools.

These are drop-in equivalents of the Milestone 1 tools. If you already have your
own Milestone 1 module, either replace the bodies below or re-export your functions
under the same names — the agent only depends on the names and signatures.

Supported resume formats: .pdf, .docx, .txt, .md. Uploaded resumes live in data/uploaded_resumes/.
"""
from __future__ import annotations

import fnmatch
import io
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from skills import Resume, parse_resume

BASE_DIR = Path(__file__).resolve().parent
RESUME_DIR = Path(os.environ.get("RESUME_DIR", BASE_DIR / "data" / "resumes"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "data" / "uploaded_resumes"))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", BASE_DIR / "reports"))
SUPPORTED = (".txt", ".md", ".pdf", ".docx")
MIN_TEXT_CHARS = 80  # below this a PDF is almost certainly scanned images, not text

# Which folders make up the candidate pool. The CLI uses RESUME_DIR only; the Streamlit app
# can add UPLOAD_DIR (and optionally drop the samples) with configure_sources().
SOURCES = {"resumes": True, "uploads": True}
# file name -> problem, for files that could not be read (shown in the UI)
LOAD_WARNINGS: dict[str, str] = {}


def _clear_caches() -> None:
    load_all_resumes.cache_clear()
    _read_text.cache_clear()
    try:  # the RAG index is built from the resume pool, so rebuild it next time it's used
        import rag_tools
        rag_tools.RETRIEVER = None
    except ImportError:  # pragma: no cover
        pass


def set_resume_dir(path: str | Path) -> None:
    """Point the tools at a different resume folder (clears caches)."""
    global RESUME_DIR
    RESUME_DIR = Path(path)
    _clear_caches()


def configure_sources(include_samples: bool = True, include_uploads: bool = True) -> None:
    SOURCES["resumes"], SOURCES["uploads"] = include_samples, include_uploads
    _clear_caches()


# ---------------------------------------------------------------------------
# Plain helpers (used directly by the agent for speed; wrapped as tools below)
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60] or "resume"


def _resume_paths() -> list[tuple[str, Path]]:
    """(candidate_id, path) for every resume in the active folders."""
    out = []
    folders = []
    if SOURCES["resumes"]:
        folders.append((RESUME_DIR, ""))
    if SOURCES["uploads"] and UPLOAD_DIR.resolve() != RESUME_DIR.resolve():
        folders.append((UPLOAD_DIR, "up_"))
    for folder, prefix in folders:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED:
                    out.append((prefix + (_slug(p.stem) if prefix else p.stem), p))
    return out


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Text from an uploaded file's bytes (.pdf, .docx, .txt, .md)."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("Install pypdf to read PDF files:  python -m pip install pypdf") from e
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise RuntimeError("PDF is password-protected") from e
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:
            raise RuntimeError("Install python-docx to read Word files:  python -m pip install python-docx") from e
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:  # many resume templates lay out sections in tables
            for row in table.rows:
                parts.append("  ".join(dict.fromkeys(c.text.strip() for c in row.cells if c.text.strip())))
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix} (use PDF, DOCX, TXT or MD)")


@lru_cache(maxsize=1024)
def _read_text(path_str: str) -> str:
    path = Path(path_str)
    return extract_text_from_bytes(path.name, path.read_bytes())


@lru_cache(maxsize=1)
def load_all_resumes() -> dict[str, Resume]:
    """Parse every resume in the active folders. A bad file is skipped (see LOAD_WARNINGS), never fatal."""
    out: dict[str, Resume] = {}
    LOAD_WARNINGS.clear()
    for cid, p in _resume_paths():
        try:
            text = _read_text(str(p))
        except Exception as e:  # corrupt / encrypted / missing library
            LOAD_WARNINGS[p.name] = str(e)
            continue
        if len(text.strip()) < MIN_TEXT_CHARS:
            LOAD_WARNINGS[p.name] = "no readable text (scanned image PDF?) — export it as a text PDF or DOCX"
            continue
        out[cid] = parse_resume(text, candidate_id=cid, source_path=str(p))
    return out


def save_uploaded_resume(filename: str, data: bytes) -> tuple[str, Resume | None, str]:
    """Store an uploaded resume and parse it. Returns (candidate_id, resume or None, warning)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        return "", None, f"unsupported file type {suffix}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = _slug(Path(filename).stem)
    target = UPLOAD_DIR / f"{stem}{suffix}"
    if not target.exists() or target.read_bytes() != data:
        target.write_bytes(data)
        _clear_caches()
    cid = "up_" + stem
    try:
        text = extract_text_from_bytes(filename, data)
    except Exception as e:
        return cid, None, str(e)
    if len(text.strip()) < MIN_TEXT_CHARS:
        return cid, None, "no readable text (scanned image PDF?) — export it as a text PDF or DOCX"
    return cid, parse_resume(text, candidate_id=cid, source_path=str(target)), ""


def list_uploaded_files() -> list[Path]:
    return sorted(p for p in UPLOAD_DIR.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED) if UPLOAD_DIR.exists() else []


def clear_uploads() -> None:
    for p in list_uploaded_files():
        p.unlink()
    _clear_caches()


def get_resume(candidate_id: str) -> Resume:
    resumes = load_all_resumes()
    if candidate_id not in resumes:
        raise KeyError(f"Unknown candidate_id '{candidate_id}'")
    return resumes[candidate_id]


def read_text_file(path: str) -> str:
    """Read a JD or any document (.txt/.md/.pdf/.docx; relative paths resolve against the project)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = BASE_DIR / p
    return _read_text(str(p))


def save_report(filename: str, content: str) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    n = 2
    while path.exists():  # never overwrite an earlier report from the same second
        path = REPORT_DIR / f"{Path(filename).stem}_{n}{Path(filename).suffix}"
        n += 1
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# LangChain tools (exposed to the LLM / ToolNode)
# ---------------------------------------------------------------------------
@tool
def list_resumes(pattern: str = "*") -> list[str]:
    """List candidate IDs (resume file stems) in the resume folder, optionally filtered by a glob pattern."""
    return [cid for cid, p in _resume_paths() if fnmatch.fnmatch(p.name, pattern) or fnmatch.fnmatch(cid, pattern)]


@tool
def read_resume(candidate_id: str) -> str:
    """Return the full raw text of one candidate's resume."""
    return get_resume(candidate_id).raw_text


@tool
def get_resume_metadata(candidate_id: str) -> str:
    """Return parsed, structured fields of a resume (name, title, years, skills, roles, education) as JSON."""
    return json.dumps(get_resume(candidate_id).to_dict(), indent=2, default=str)


@tool
def search_files(keyword: str, max_results: int = 20) -> list[dict]:
    """Case-insensitive keyword (exact substring) search across all resume files. Returns matching lines."""
    hits = []
    kw = keyword.lower()
    for cid, r in load_all_resumes().items():
        for line in r.raw_text.splitlines():
            if kw in line.lower():
                hits.append({"candidate_id": cid, "name": r.name, "line": line.strip()})
                break
        if len(hits) >= max_results:
            break
    return hits


@tool
def write_report(filename: str, content: str) -> str:
    """Write a markdown report into the reports folder and return its path."""
    return save_report(filename, content)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


FS_TOOLS = [list_resumes, read_resume, get_resume_metadata, search_files, write_report]
