"""The 6 documented conversation flows from scenarios.py, run as tests."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("MATCHER_LLM", None)
os.environ["UPLOAD_DIR"] = __import__("tempfile").mkdtemp(prefix="uploads_")  # ignore real uploads
os.environ.setdefault("REPORT_DIR", tempfile.mkdtemp(prefix="match_reports_"))

import generate_sample_data  # noqa: E402

generate_sample_data.main()

from scenarios import FLOWS, run_flow  # noqa: E402


@pytest.mark.parametrize("flow", FLOWS, ids=[f["id"] for f in FLOWS])
def test_conversation_flow(flow):
    _, failures = run_flow(flow)
    assert not failures, "\n".join(failures)
