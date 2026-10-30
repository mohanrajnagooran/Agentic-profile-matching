"""
scenarios.py — the 6 documented conversation flows (submission item: "Test scenarios: 5+ conversation flows").

    python scenarios.py          # runs every flow, checks it, writes docs/TEST_SCENARIOS.md
    pytest -q                    # tests/test_conversation_flows.py runs the same flows as tests

Each flow is a list of turns: (user message, [substrings the agent's reply must contain]).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
JD = str(BASE / "data" / "job_descriptions" / "senior_fullstack.txt")

FLOWS = [
    {
        "id": "F1",
        "title": "JD screening → compare → explain",
        "covers": "Part A pipeline, compare_candidates, explainability",
        "start": {"jd_path": JD},
        "start_expect": ["Top 10", "Must-have:", "REST APIs"],
        "turns": [
            ("Compare the top 3 matches side by side", ["Best overall fit", "Must: React", "Nice-to-haves"]),
            ("Why did John rank higher than Jane?", ["John Smith", "Jane Doe", "Decisive factors", "What would flip it"]),
            ("Why is #2 ranked there?", ["Here's what separates them"]),
            ("done", ["Done."]),
        ],
    },
    {
        "id": "F2",
        "title": "Natural-language search, then refine",
        "covers": "Conversational search, iterative refinement, re-rank explanations",
        "start": {"query": "Find me candidates with React and 3+ years experience"},
        "start_expect": ["Top 10", "Must-have:** React", "3+ yrs"],
        "turns": [
            ("Make TypeScript a must-have", ["Re-ranked after", "TypeScript → must-have", "Ranking changes"]),
            ("AWS is a plus", ["AWS → nice-to-have"]),
            ("Explain the ranking changes", ["Why the ranking changed"]),
            ("Show requirements", ["Changes this session", "TypeScript"]),
            ("done", ["Done."]),
        ],
    },
    {
        "id": "F3",
        "title": "Adjust requirements mid-conversation (several at once) and reset",
        "covers": "Refinement: must→nice, drop, years, weights, location, reset",
        "start": {"jd_path": JD},
        "start_expect": ["Top 10"],
        "turns": [
            ("Make AWS a must-have", ["AWS → must-have", "newly required AWS"]),
            ("Why did John rank higher than Jane?", ["John Smith", "Jane Doe"]),
            ("Docker is optional; drop GraphQL, require 7+ years and prioritize experience",
             ["removed GraphQL", "overall experience → 7+ yrs", "increased weight on experience"]),
            ("Only candidates in Bengaluru", ["location filter → Bengaluru"]),
            ("Reset to the original JD", ["reset to the original job requirements"]),
            ("done", ["Done."]),
        ],
    },
    {
        "id": "F4",
        "title": "Multi-round screening → hire / no-hire",
        "covers": "Part C: round 1 top-10 of 100, round 2 deep analysis, round 3 recommendations",
        "start": {"jd_path": JD},
        "start_expect": ["screened 100 resumes"],
        "turns": [
            ("Run multi-round screening", ["Round 1", "top 10 advance", "Round 2", "Round 3", "Hire"]),
            ("Tell me about #1", ["Match report", "Strengths", "Gaps", "Recommendation:"]),
            ("done", ["Done."]),
        ],
    },
    {
        "id": "F5",
        "title": "Borderline candidate: explainability + interview prep",
        "covers": "Detailed match report, strengths/gaps, improvement suggestions, generate_interview_questions",
        "start": {"jd_path": JD},
        "start_expect": ["Top 10"],
        "turns": [
            ("Tell me about Jane Doe", ["Strengths", "Gaps", "Missing must-have: TypeScript", "Improvement suggestions"]),
            ("Interview questions for Jane Doe", ["Gap probe", "TypeScript", "Behavioural"]),
            ("Compare John and Jane", ["John Smith", "Jane Doe", "Meets every must-have"]),
            ("done", ["Done."]),
        ],
    },
    {
        "id": "F6",
        "title": "Robustness: help, unclear input, ambiguous names",
        "covers": "Graceful handling of unrecognised input; agent stays in the loop",
        "start": {"jd_path": JD},
        "start_expect": ["Top 10"],
        "turns": [
            ("help", ["Here's what I can do"]),
            ("what's the weather like?", ["didn't catch"]),
            ("make it better somehow", ["didn't catch"]),
            ("Tell me about Priya", ["matches", "Match report"]),
            ("Show the top 5", ["Current shortlist"]),
            ("done", ["Done."]),
        ],
    },
]


def run_flow(flow: dict) -> tuple[list[tuple[str, str]], list[str]]:
    """Run one flow. Returns (transcript, failures)."""
    from matching_agent import MatchingAgent

    agent = MatchingAgent()
    transcript, failures = [], []
    label = flow["start"].get("query") or f"[job description: {Path(flow['start']['jd_path']).name}]"
    reply = agent.start(**flow["start"])
    transcript.append((label, reply))
    failures += [f"start: missing '{e}'" for e in flow["start_expect"] if e not in reply]
    for msg, expect in flow["turns"]:
        reply = agent.chat(msg)
        transcript.append((msg, reply))
        failures += [f"'{msg}': missing '{e}'" for e in expect if e not in reply]
    if not agent.finished:
        failures.append("conversation did not reach END")
    return transcript, failures


def main() -> int:
    os.environ.setdefault("REPORT_DIR", str(BASE / "reports"))
    out = ["# Test scenarios — conversation flows", "",
           "Generated by `python scenarios.py`. The same flows run as automated tests in `tests/test_conversation_flows.py`.", "",
           "| Flow | Scenario | Covers | Turns | Result |", "|---|---|---|---|---|"]
    bodies, total_fail = [], 0
    for f in FLOWS:
        transcript, failures = run_flow(f)
        total_fail += len(failures)
        status = "✅ pass" if not failures else f"❌ {len(failures)} failed"
        out.append(f"| {f['id']} | {f['title']} | {f['covers']} | {len(transcript)} | {status} |")
        body = [f"\n---\n\n## {f['id']} — {f['title']}", f"_Covers: {f['covers']}_", ""]
        for msg, reply in transcript:
            body += [f"**👤 User:** {msg}", "", "**🤖 Agent:**", "", reply, ""]
        if failures:
            body += ["**Failures:**", *[f"- {x}" for x in failures]]
        bodies.append("\n".join(body))
        print(f"{f['id']} {status} — {f['title']}")
    path = BASE / "docs" / "TEST_SCENARIOS.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n" + "\n".join(bodies), encoding="utf-8")
    print(f"Wrote {path}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
