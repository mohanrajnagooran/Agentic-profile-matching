"""PDF / DOCX resume uploads: parsing real-world layouts, ranking, and bad files."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("MATCHER_LLM", None)
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="uploads_")
os.environ.setdefault("REPORT_DIR", tempfile.mkdtemp(prefix="match_reports_"))

import generate_sample_data  # noqa: E402

generate_sample_data.main()

import fs_tools  # noqa: E402
from matching_agent import MatchingAgent  # noqa: E402

SAMPLES = ROOT / "data" / "sample_uploads"
JD = ROOT / "data" / "job_descriptions" / "senior_fullstack.txt"

EXPECTED = {
    # file: (name, current title, company, min total years, skills that must be found)
    "Arjun_Mehta_Resume.pdf": ("Arjun Mehta", "Senior Software Engineer", "Razorpay", 8, {"React", "TypeScript", "Node.js", "REST APIs", "AWS"}),
    "Priya_Raman_CV.pdf": ("Priya Raman", "Frontend Engineer", "Swiggy", 6, {"React", "TypeScript", "Angular", "Cypress"}),
    "Karan_Shah_Fresher.pdf": ("Karan Shah", "Backend Intern", "Postman", 0, {"Python", "FastAPI", "Docker"}),
    "Meera_Iyer_Resume.docx": ("Meera Iyer", "Lead Engineer", "Flipkart", 11, {"Java", "Spring Boot", "Kafka", "Kubernetes"}),
}


@pytest.fixture(autouse=True)
def clean_uploads():
    fs_tools.clear_uploads()
    fs_tools.configure_sources(True, True)
    yield
    fs_tools.clear_uploads()


@pytest.mark.parametrize("fname", EXPECTED)
def test_parses_real_world_layouts(fname):
    name, title, company, years, skills = EXPECTED[fname]
    cid, r, warning = fs_tools.save_uploaded_resume(fname, (SAMPLES / fname).read_bytes())
    assert r is not None, warning
    assert cid.startswith("up_")
    assert r.name == name
    current = next(x for x in r.roles if x.current) if any(x.current for x in r.roles) else r.roles[0]
    assert (current.title, current.company) == (title, company)
    assert r.total_years >= years
    assert skills <= r.all_skills()
    assert r.email and "@" in r.email
    assert r.highest_degree() in ("Bachelors", "Masters")


def test_month_level_years():
    _, r, _ = fs_tools.save_uploaded_resume("p.pdf", (SAMPLES / "Priya_Raman_CV.pdf").read_bytes())
    tcs = next(x for x in r.roles if "Tata" in x.company)
    assert tcs.years == pytest.approx(2.7, abs=0.1)  # 08/2020 - 03/2023 = 32 months
    assert r.skill_years("Angular") == pytest.approx(2.7, abs=0.1)


def test_uploads_are_ranked_with_samples():
    for f in EXPECTED:
        fs_tools.save_uploaded_resume(f, (SAMPLES / f).read_bytes())
    assert len(fs_tools.load_all_resumes()) == 104
    a = MatchingAgent()
    out = a.start(jd_text=JD.read_text())
    assert "screened 104 resumes" in out
    # Arjun meets every must-have (React 3+, TypeScript, Node.js, REST APIs) with 8+ years -> top of the list
    assert a.state["ranking"][0] == "up_arjun_mehta_resume"
    report = a.chat("Tell me about Arjun Mehta")
    assert "Arjun Mehta" in report and "Strengths" in report
    cmp = a.chat("Compare Arjun Mehta and John Smith")
    assert "Arjun Mehta" in cmp and "John Smith" in cmp


def test_only_uploads_pool():
    for f in EXPECTED:
        fs_tools.save_uploaded_resume(f, (SAMPLES / f).read_bytes())
    fs_tools.configure_sources(include_samples=False, include_uploads=True)
    try:
        a = MatchingAgent()
        out = a.start(query="Find me candidates with Java and Kafka")
        assert "screened 4 resumes" in out
        assert a.state["ranking"][0] == "up_meera_iyer_resume"
    finally:
        fs_tools.configure_sources(True, True)


def test_bad_files_are_reported_not_fatal():
    cid, r, warning = fs_tools.save_uploaded_resume("broken.pdf", b"%PDF-1.4 this is not really a pdf")
    assert r is None and warning
    cid, r, warning = fs_tools.save_uploaded_resume("photo.png", b"\x89PNG")
    assert r is None and "unsupported" in warning
    # a PDF with no text layer (e.g. a scan)
    from pypdf import PdfWriter
    import io
    w = PdfWriter(); w.add_blank_page(width=595, height=842); buf = io.BytesIO(); w.write(buf)
    cid, r, warning = fs_tools.save_uploaded_resume("scan.pdf", buf.getvalue())
    assert r is None and "scanned" in warning
    resumes = fs_tools.load_all_resumes()  # loading the pool still works
    assert len(resumes) == 100 and "broken.pdf" in fs_tools.LOAD_WARNINGS and "scan.pdf" in fs_tools.LOAD_WARNINGS
