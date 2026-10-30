"""End-to-end tests for the matching agent. Run: pytest -q"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tempfile  # noqa: E402

os.environ.pop("MATCHER_LLM", None)
os.environ["UPLOAD_DIR"] = __import__("tempfile").mkdtemp(prefix="uploads_")  # ignore real uploads  # tests are deterministic, no LLM
os.environ["REPORT_DIR"] = tempfile.mkdtemp(prefix="match_reports_")  # keep test reports out of ./reports

import generate_sample_data  # noqa: E402

generate_sample_data.main()

from matching_agent import (MatchingAgent, apply_refinement, build_graph, classify_intent,  # noqa: E402
                            compare_candidates, extract_requirements, generate_interview_questions, AGENT_TOOLS)
from scoring import parse_requirements  # noqa: E402

JD = str(ROOT / "data/job_descriptions/senior_fullstack.txt")


@pytest.fixture()
def agent():
    a = MatchingAgent()
    a.start(jd_path=JD)
    return a


# ---------------- Part A: architecture ----------------
def test_graph_has_required_pipeline():
    g = build_graph().get_graph()
    nodes = set(g.nodes)
    for n in ["parse_jd", "extract_requirements", "search_resumes", "rank_candidates", "generate_report", "human_feedback"]:
        assert n in nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "parse_jd") in edges
    assert ("generate_report", "human_feedback") in edges
    assert ("human_feedback", "__end__") in edges


def test_state_tracks_history_requirements_shortlist(agent):
    s = agent.state
    assert s["requirements"]["must_have"] and s["requirements"]["nice_to_have"]
    assert len(s["shortlist"]) == 10 and all(x["reasoning"] for x in s["shortlist"])
    assert len(s["all_scores"]) == 100
    agent.chat("Make AWS a must-have")
    s = agent.state
    assert len(s["requirement_history"]) == 2
    assert any(m.type == "human" for m in s["messages"]) and any(m.type == "ai" for m in s["messages"])


def test_tools_registered():
    names = {t.name for t in AGENT_TOOLS}
    assert {"list_resumes", "read_resume", "search_files", "write_report", "rag_search", "extract_requirements",
            "compare_candidates", "generate_interview_questions"} <= names


def test_extract_requirements_must_vs_nice():
    r = extract_requirements.invoke({"jd": open(JD).read()})
    must = {m["skill"]: m["min_years"] for m in r["must_have"]}
    nice = {n["skill"] for n in r["nice_to_have"]}
    assert must == {"React": 3.0, "TypeScript": None, "Node.js": None, "REST APIs": None}
    assert nice == {"AWS", "Docker", "GraphQL", "PostgreSQL", "Kubernetes"}
    assert r["min_years"] == 5 and r["education"]["level"] == "Bachelors" and not r["education"]["required"]


def test_nl_query_requirements():
    r = parse_requirements("React and 3+ years experience")
    assert [m["skill"] for m in r["must_have"]] == ["React"] and r["min_years"] == 3
    r = parse_requirements("React, TypeScript and 5+ years of Python, AWS is a plus")
    assert {m["skill"]: m["min_years"] for m in r["must_have"]}["Python"] == 5
    assert [n["skill"] for n in r["nice_to_have"]] == ["AWS"]


# ---------------- Part B: conversational ----------------
@pytest.mark.parametrize("text,intent", [
    ("Find me candidates with React and 3+ years experience", "search"),
    ("Compare the top 3 matches side by side", "compare"),
    ("Why did John rank higher than Jane?", "explain"),
    ("Make AWS a must-have", "refine"),
    ("Only candidates in Chennai", "refine"),
    ("Run multi-round screening", "screen"),
    ("Interview questions for #1", "questions"),
    ("Tell me about Jane Doe", "detail"),
    ("done", "done"),
])
def test_intents(text, intent):
    assert classify_intent(text) == intent


def test_search_query_without_jd():
    a = MatchingAgent()
    out = a.start(query="Find me candidates with React and 3+ years experience")
    assert "Top 10" in out
    s = a.state
    assert [m["skill"] for m in s["requirements"]["must_have"]] == ["React"]
    assert all(s["all_scores"][x["candidate_id"]]["total_years"] >= 3 for x in s["shortlist"])


def test_compare_top3(agent):
    out = agent.chat("Compare the top 3 matches side by side")
    top3 = [agent.state["all_scores"][c]["name"] for c in agent.state["ranking"][:3]]
    assert all(n in out for n in top3) and "Best overall fit" in out


def test_why_john_vs_jane(agent):
    out = agent.chat("Why did John rank higher than Jane?")
    assert "John Smith" in out and "Jane Doe" in out and "Decisive factors" in out


def test_refinement_reranks_and_explains(agent):
    before = agent.state["ranking"][:]
    out = agent.chat("Make AWS a must-have")
    after = agent.state["ranking"]
    assert before != after
    assert "Ranking changes" in out and "AWS" in out
    assert any(m["skill"] == "AWS" for m in agent.state["requirements"]["must_have"])
    # Jane has AWS, John does not -> Jane moves up, John moves down
    assert after.index("cand_002") < before.index("cand_002")
    assert after.index("cand_001") > before.index("cand_001")
    out = agent.chat("Why did John rank higher than Jane?")
    assert "John Smith" in out and "Jane Doe" in out


def test_refinement_operations():
    base = parse_requirements(open(JD).read())
    new, ch = apply_refinement(base, "Docker is optional; drop GraphQL, require 7+ years and prioritize experience")
    assert ch
    assert new["min_years"] == 7
    assert "GraphQL" not in {n["skill"] for n in new["nice_to_have"]}
    assert new["weights"]["experience"] > base["weights"]["experience"]
    new, _ = apply_refinement(base, "TypeScript is nice to have")
    assert "TypeScript" not in {m["skill"] for m in new["must_have"]}
    assert "TypeScript" in {n["skill"] for n in new["nice_to_have"]}
    new, _ = apply_refinement(base, "require 5+ years of Node.js")
    assert {m["skill"]: m["min_years"] for m in new["must_have"]}["Node.js"] == 5


def test_reset_restores_original(agent):
    orig = agent.state["ranking"][:10]
    agent.chat("Make AWS a must-have")
    agent.chat("Reset to the original JD")
    assert agent.state["ranking"][:10] == orig


def test_location_filter(agent):
    agent.chat("Only candidates in Chennai")
    s = agent.state
    assert s["shortlist"] and all("Chennai" in s["all_scores"][x["candidate_id"]]["location"] for x in s["shortlist"])


def test_unparseable_refinement_is_handled(agent):
    out = agent.chat("hmm what about the weather")
    assert "didn't catch" in out or "relevant" in out
    assert not agent.finished


# ---------------- Part C: advanced ----------------
def test_multi_round_screening(agent):
    out = agent.chat("Run multi-round screening")
    sc = agent.state["screening"]
    assert len(sc["round1"]) == 10 and set(sc["final_order"]) == set(sc["round1"])
    verdicts = {d["verdict"] for d in sc["decisions"].values()}
    assert verdicts <= {"Strong Hire", "Hire", "Borderline (Lean No Hire)", "No Hire"}
    assert "Round 1" in out and "Round 2" in out and "Round 3" in out
    report = Path(sc["report_path"]).read_text()
    assert "Round 2 — deep analysis" in report and "Strengths" in report and "Gaps" in report


def test_match_report_strengths_gaps_suggestions(agent):
    out = agent.chat("Tell me about Jane Doe")
    assert "Strengths" in out and "Gaps" in out and "Improvement suggestions" in out
    assert "TypeScript" in out  # Jane's real gap


def test_interview_questions(agent):
    out = agent.chat("Interview questions for Jane Doe")
    assert "Gap probe" in out and "TypeScript" in out and "Behavioural" in out


def test_tools_standalone(agent):
    out = compare_candidates.invoke({"candidate_ids": ["cand_001", "cand_002"]})
    assert "John Smith" in out and "Jane Doe" in out
    q = generate_interview_questions.invoke({"candidate_id": "cand_001"})
    assert "Screening questions for John Smith" in q


def test_done_ends_graph(agent):
    agent.chat("done")
    assert agent.finished


def test_free_form_question_uses_llm_tools(agent):
    """With an LLM configured, unrecognised questions go to a tool-calling loop over the agent's tools."""
    import matching_agent
    from langchain_core.messages import AIMessage

    class StubLLM:
        def __init__(self):
            self.calls = 0
        def bind_tools(self, tools):
            assert {t.name for t in tools} >= {"rag_search", "compare_candidates"}
            return self
        def invoke(self, msgs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage("", tool_calls=[{"name": "rag_search", "args": {"query": "Kafka", "k": 3}, "id": "t1"}])
            assert msgs[-1].type == "tool" and "candidate_id" in msgs[-1].content
            return AIMessage("Three candidates mention Kafka.")

    matching_agent._LLM = StubLLM()
    try:
        out = agent.chat("Which of these people would survive a Kafka outage at 3am?")
        assert "Three candidates mention Kafka." in out
    finally:
        matching_agent._LLM = None
