"""
app.py — Streamlit chat interface for the matching agent.

    streamlit run app.py

The same LangGraph agent as the CLI: the sidebar starts a session from a job description
or a natural-language search; the chat box feeds the human-feedback loop.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import fs_tools
from fs_tools import load_all_resumes
from matching_agent import HELP_TEXT, MatchingAgent
from scoring import requirements_summary

BASE = Path(__file__).resolve().parent
JD_DIR = BASE / "data" / "job_descriptions"
DIAGRAM = BASE / "docs" / "state_machine.png"

st.set_page_config(page_title="Resume Matching Agent", page_icon="🧭", layout="wide")

if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.chat = []  # list of (role, markdown)


def start_session(jd_text: str | None, query: str | None) -> None:
    agent = MatchingAgent()
    with st.spinner("Parsing requirements, searching and ranking resumes…"):
        reply = agent.start(jd_text=jd_text, query=query)
    st.session_state.agent = agent
    st.session_state.pool = pool_signature()
    st.session_state.chat = [("user", query or "Match candidates to this job description."), ("assistant", reply)]


def send(message: str) -> None:
    agent: MatchingAgent = st.session_state.agent
    st.session_state.chat.append(("user", message))
    with st.spinner("Thinking…"):
        reply = agent.chat(message)
    st.session_state.chat.append(("assistant", reply))


def pool_signature() -> tuple:
    return tuple(sorted(load_all_resumes()))


def handle_uploads(files) -> None:
    """Save newly uploaded resumes (PDF/DOCX/TXT/MD) and remember how each one parsed."""
    seen = st.session_state.setdefault("processed_uploads", {})
    for f in files or []:
        key = (f.name, f.size)
        if key in seen:
            continue
        cid, resume, warning = fs_tools.save_uploaded_resume(f.name, f.getvalue())
        seen[key] = {"file": f.name, "id": cid, "ok": resume is not None, "warning": warning}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 Matching Agent")

    # ---- 1. resumes
    st.subheader("1 · Resumes")
    st.session_state.setdefault("uploader_key", 0)
    files = st.file_uploader("Upload resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, key=f"resumes_{st.session_state.uploader_key}")
    handle_uploads(files)
    uploaded = fs_tools.list_uploaded_files()
    pool_choice = st.radio("Candidate pool", ["Sample resumes + my uploads", "Only my uploads"], horizontal=False,
                           disabled=not uploaded, help="The 100 sample resumes are in data/resumes; uploads in data/uploaded_resumes")
    want = (pool_choice == "Sample resumes + my uploads" or not uploaded, True)
    if (fs_tools.SOURCES["resumes"], fs_tools.SOURCES["uploads"]) != want:
        fs_tools.configure_sources(*want)

    resumes = load_all_resumes()
    n_up = sum(1 for c in resumes if c.startswith("up_"))
    st.caption(f"**{len(resumes)}** resumes in the pool ({n_up} uploaded)")
    if uploaded:
        with st.expander(f"Uploaded resumes ({len(uploaded)}) — check they parsed correctly", expanded=bool(files)):
            rows = []
            for cid, r in sorted(((c, r) for c, r in resumes.items() if c.startswith("up_")), key=lambda x: x[1].name):
                rows.append({"Name": r.name, "Years": r.total_years, "Roles": len(r.roles),
                             "Skills": ", ".join(sorted(r.all_skills())[:8]) + ("…" if len(r.all_skills()) > 8 else ""),
                             "File": Path(r.source_path).name})
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            for name, problem in fs_tools.LOAD_WARNINGS.items():
                st.warning(f"**{name}** skipped: {problem}")
            if st.button("🗑 Remove all uploaded resumes", width="stretch"):
                fs_tools.clear_uploads()
                st.session_state.processed_uploads = {}
                st.session_state.uploader_key += 1
                st.rerun()
    if not resumes:
        st.error("No resumes yet. Upload some above, or run `python generate_sample_data.py` for the 100 samples.")
        st.stop()

    # ---- 2. job
    st.subheader("2 · Job")
    mode = st.radio("Start from", ["Job description", "Natural-language search"], horizontal=True)
    if mode == "Job description":
        samples = sorted(p.name for p in JD_DIR.glob("*.txt"))
        choice = st.selectbox("Sample JD", ["(paste or upload)"] + samples, index=1 if samples else 0)
        default = (JD_DIR / choice).read_text() if choice in samples else ""
        upload = st.file_uploader("…or upload a JD (PDF, DOCX, TXT)", type=["txt", "md", "pdf", "docx"])
        if upload:
            try:
                default = fs_tools.extract_text_from_bytes(upload.name, upload.getvalue())
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")
        jd_text = st.text_area("Job description", value=default, height=260)
        if st.button("Start matching", type="primary", width="stretch", disabled=not jd_text.strip()):
            start_session(jd_text, None)
    else:
        q = st.text_input("Search", "Find me candidates with React and 3+ years experience")
        if st.button("Search", type="primary", width="stretch", disabled=not q.strip()):
            start_session(None, q)

    if st.session_state.agent and st.session_state.get("pool") != pool_signature():
        st.info("The resume pool changed. Press **Start matching** again to rank the new resumes.")

    agent: MatchingAgent | None = st.session_state.agent
    if agent and agent.state.get("requirements"):
        s = agent.state
        st.divider()
        st.subheader("Current requirements")
        st.markdown(requirements_summary(s["requirements"], multiline=True))
        st.subheader("Shortlist")
        st.dataframe([{"#": x["rank"], "Candidate": x["name"], "Score": x["score"]} for x in s.get("shortlist", [])],
                     hide_index=True, width="stretch")
        if s.get("last_report_path") and Path(s["last_report_path"]).exists():
            p = Path(s["last_report_path"])
            st.download_button("⬇ Download latest report", p.read_text(), file_name=p.name, width="stretch")
        with st.expander("Agent reasoning log"):
            for line in s.get("reasoning_log", [])[-25:]:
                st.text(line)
    if DIAGRAM.exists():
        with st.expander("State machine"):
            st.image(str(DIAGRAM))

# ---------------------------------------------------------------- main chat
agent = st.session_state.agent
if agent is None:
    st.header("Resume ↔ Job Matching Agent")
    st.markdown("**1.** Upload resumes (PDF, DOCX or TXT) in the sidebar, or use the 100 sample resumes.  \n"
                "**2.** Pick or upload a job description, or type a search, then press **Start matching**.\n\n" + HELP_TEXT)
    st.stop()

QUICK = ["Compare the top 3 matches side by side", "Why did John rank higher than Jane?", "Make AWS a must-have",
         "Run multi-round screening", "Tell me about Jane Doe", "Interview questions for #1", "Reset to the original JD"]
if not agent.finished:
    cols = st.columns(4)
    for i, q in enumerate(QUICK):
        if cols[i % 4].button(q, key=f"quick{i}", width="stretch"):
            send(q)

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if agent.finished:
    st.info("Session ended. Start a new one from the sidebar.")
elif prompt := st.chat_input("Ask the agent… (e.g. 'Make Docker a must-have', 'why is #4 ranked there?')"):
    send(prompt)
    st.rerun()
