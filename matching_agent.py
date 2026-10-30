"""
matching_agent.py — LangGraph resume ↔ job matching agent (Milestone 3).

Graph
-----
    START → parse_jd → extract_requirements → search_resumes → rank_candidates
          → generate_report → human_feedback ─┬─► END                         ("done")
                                               ├─► extract_requirements        (new natural-language search)
                                               ├─► refine_requirements ──► search_resumes   (adjust criteria → re-rank)
                                               ├─► search_resumes              (reset to original JD)
                                               ├─► compare_node                (head-to-head)
                                               ├─► explain_node                ("why did John rank higher than Jane?")
                                               ├─► interview_node              (screening questions)
                                               ├─► candidate_report_node       (strengths / gaps / suggestions)
                                               ├─► screening_node              (3-round screen → hire / no-hire)
                                               ├─► status_node                 (show shortlist / requirements)
                                               └─► tool_agent                  (free-form; LLM + tools if configured)
          every handler returns to human_feedback (the human-in-the-loop interrupt).

Usage
-----
    python generate_sample_data.py                      # once: 100 sample resumes + a JD
    python matching_agent.py --jd data/job_descriptions/senior_fullstack.txt
    python matching_agent.py --query "Find me candidates with React and 3+ years experience"
    python matching_agent.py --demo                     # scripted walkthrough of every feature

Optional LLM: set MATCHER_LLM (e.g. "anthropic:<model-name>" or "openai:<model-name>") and install
`langchain` + the provider package. The LLM is only used for free-form questions the rule-based
router doesn't recognise; all scoring stays deterministic and explainable.
"""
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

import fs_tools
from fs_tools import FS_TOOLS, get_resume, load_all_resumes, read_text_file, save_report, timestamp
from rag_tools import RAG_TOOLS, get_retriever, semantic_search
from scoring import (COMPONENT_LABELS, DEFAULT_WEIGHTS, NICE_MARK, YEARS_RE, deep_profile, deep_score,
                     explain_pair, explain_rank_changes, fmt_years, hire_decision, improvement_suggestions,
                     one_line_reason, parse_requirements, requirements_summary, score_candidate, split_clauses,
                     strengths_and_gaps, user_skills)
from skills import SKILL_TAXONOMY

TOP_K = 10          # shortlist size (round-1 output)
SEARCH_K = 100      # how many resumes the RAG search pulls into the pool
FINAL_K = 10        # how many go through deep analysis


# =============================================================================
# 1. Agent state
# =============================================================================
class AgentState(TypedDict, total=False):
    # conversation
    messages: Annotated[list[AnyMessage], add_messages]
    last_user_message: str
    intent: str
    # job understanding
    jd_text: str
    requirements_source: str               # text requirements are extracted from (JD or a NL query)
    requirements_mode: str                 # "jd" | "query"
    requirements: dict                     # current must/nice/years/education/location/weights
    original_requirements: dict            # from the JD — used by "reset"
    previous_requirements: dict
    requirement_history: Annotated[list[dict], operator.add]
    change_note: str
    # candidates
    candidate_pool: list[dict]             # RAG hits: id, name, relevance, snippet
    all_scores: dict[str, dict]            # id -> explainable score breakdown
    previous_scores: dict[str, dict]
    ranking: list[str]                     # every eligible candidate, best first
    previous_ranking: list[str]
    shortlist: list[dict]                  # top-K: rank, id, name, score, reasoning
    rank_changes: list[dict]
    screening: dict                        # multi-round results
    reasoning_log: Annotated[list[str], operator.add]
    last_report_path: str


# Tools are plain functions the LLM can also call; they read the agent's current
# understanding of the job from this context (synced from state by every node).
CONTEXT: dict[str, Any] = {"requirements": None, "scores": {}, "ranking": []}


def _sync_context(state: AgentState) -> None:
    CONTEXT["requirements"] = state.get("requirements")
    CONTEXT["scores"] = state.get("all_scores", {})
    CONTEXT["ranking"] = state.get("ranking", [])


def _known_locations() -> list[str]:
    locs = {r.location for r in load_all_resumes().values() if r.location}
    return sorted(locs, key=len, reverse=True)


# =============================================================================
# 2. Additional tools
# =============================================================================
@tool
def extract_requirements(jd: str) -> dict:
    """Parse a job description (or a natural-language search) into must-have vs nice-to-have requirements:
    skills (with per-skill minimum years), overall minimum years, education and location."""
    return parse_requirements(jd, _known_locations())


def _score_for(cid: str) -> dict:
    """Score breakdown for a candidate under the current requirements (computed on demand if needed)."""
    if cid in CONTEXT["scores"]:
        return CONTEXT["scores"][cid]
    reqs = CONTEXT["requirements"] or parse_requirements("")
    return score_candidate(get_resume(cid), reqs, relevance=0.0)


def _rank_of(cid: str) -> int | None:
    try:
        return CONTEXT["ranking"].index(cid) + 1
    except ValueError:
        return None


@tool
def compare_candidates(candidate_ids: list[str]) -> str:
    """Head-to-head comparison of 2-5 candidates against the current job requirements, as a markdown table
    with must-have coverage, experience, nice-to-haves, education, strengths, gaps and a verdict."""
    reqs = CONTEXT["requirements"] or parse_requirements("")
    ids = list(dict.fromkeys(candidate_ids))[:5]
    if len(ids) < 2:
        return "Need at least two candidates to compare."
    rs = [get_resume(c) for c in ids]
    bs = [_score_for(c) for c in ids]
    profs = [deep_profile(r, reqs) for r in rs]

    def row(label, vals):
        return f"| **{label}** | " + " | ".join(vals) + " |"

    out = ["| | " + " | ".join(f"**{b['name']}**" for b in bs) + " |", "|---|" + "---|" * len(ids)]
    out.append(row("Rank / score", [f"#{_rank_of(c) or '–'} · {b['score']}" for c, b in zip(ids, bs)]))
    out.append(row("Current role", [p["current_role"] for p in profs]))
    out.append(row("Experience", [f"{fmt_years(b['total_years'])} yrs" for b in bs]))
    for m in reqs["must_have"]:
        s = m["skill"]
        cells = []
        for r, b in zip(rs, bs):
            if s in b["missing_must"]:
                cells.append("✗")
            else:
                y = r.skill_years(s)
                short = any(x["skill"] == s for x in b["short_years"])
                cells.append(("⚠ " if short else "✓ ") + (f"{fmt_years(y)} yrs" if y else "listed"))
        out.append(row(f"Must: {s}" + (f" ({fmt_years(m['min_years'])}+)" if m.get("min_years") else ""), cells))
    if reqs["nice_to_have"]:
        out.append(row(f"Nice-to-haves ({len(reqs['nice_to_have'])})",
                       [f"{len(b['matched_nice'])}: {', '.join(b['matched_nice']) or '—'}" for b in bs]))
    out.append(row("Education", [b["degree"] for b in bs]))
    out.append(row("Quantified impact", [f"{len(p['quantified'])}/{p['n_bullets']} bullets" for p in profs]))
    out.append(row("Leadership signals", ["yes" if p["leadership"] else "no" for p in profs]))
    out.append(row("Certifications", ["; ".join(p["relevant_certs"]) or "—" for p in profs]))
    sg = [strengths_and_gaps(r, b, reqs, p) for r, b, p in zip(rs, bs, profs)]
    out.append(row("Top strength", [s[0] if s else "—" for s, _ in sg]))
    out.append(row("Main gap", [g[0] if g else "—" for _, g in sg]))

    best = max(bs, key=lambda b: b["score"])
    most_exp = max(bs, key=lambda b: b["total_years"])
    most_nice = max(bs, key=lambda b: len(b["matched_nice"]))
    summary = [f"\n**Best overall fit:** {best['name']} ({best['score']})."]
    if most_exp is not best:
        summary.append(f"**Most experienced:** {most_exp['name']} ({fmt_years(most_exp['total_years'])} yrs).")
    if reqs["nice_to_have"] and most_nice is not best and len(most_nice["matched_nice"]) > len(best["matched_nice"]):
        summary.append(f"**Broadest extras:** {most_nice['name']} ({', '.join(most_nice['matched_nice'])}).")
    complete = [b["name"] for b in bs if not b["missing_must"] and not b["short_years"]]
    summary.append(f"**Meets every must-have:** {', '.join(complete) if complete else 'nobody in this group'}.")
    return "\n".join(out) + "\n" + " ".join(summary)


QUESTION_BANK = {
    "React": ["How do you decide between local state, context and a store like Redux? Walk me through a real case.",
              "Tell me about a React performance problem you diagnosed. How did you find it and what fixed it?"],
    "TypeScript": ["Show how you've used generics or discriminated unions to make an API impossible to misuse.",
                   "How would you migrate a large JavaScript codebase to strict TypeScript incrementally?"],
    "Node.js": ["Explain the Node.js event loop. How would you handle a CPU-heavy task without blocking requests?",
                "How do you structure error handling, retries and graceful shutdown in a Node service?"],
    "REST APIs": ["Design the endpoints for issuing refunds. How do you make them idempotent and version them?"],
    "GraphQL": ["How do you prevent N+1 queries and protect a GraphQL server from expensive queries?"],
    "AWS": ["Which AWS services would you use for a highly available API, and what would you watch for cost?"],
    "Docker": ["How do you keep Docker images small, reproducible and secure?"],
    "Kubernetes": ["A pod is in CrashLoopBackOff in production. Walk me through your debugging steps."],
    "PostgreSQL": ["A query on a 50M-row Postgres table got slow. How do you investigate and fix it?"],
    "SQL": ["How would you find duplicate records and keep only the latest per customer in SQL?"],
    "Python": ["When would you reach for asyncio vs multiprocessing in Python, and why?"],
    "Django": ["How do you avoid N+1 queries in the Django ORM and keep migrations safe on a large table?"],
    "Java": ["How do you diagnose a memory leak or GC pressure in a JVM service?"],
    "Spring Boot": ["How do you structure configuration, transactions and error handling in a Spring Boot service?"],
    "Microservices": ["How do you keep data consistent across services without distributed transactions?"],
    "System Design": ["Design a rate limiter for a public API. Where does state live and how does it scale?"],
    "Machine Learning": ["How do you detect data leakage and choose a validation strategy for a new model?"],
    "Terraform": ["How do you manage Terraform state and review infrastructure changes safely in a team?"],
    "CI/CD": ["Describe a CI/CD pipeline you built. How did you keep it fast and make deploys safe to roll back?"],
    "Angular": ["How do you manage state and change detection performance in a large Angular app?"],
    "Kafka": ["How do you guarantee ordering and handle duplicate messages with Kafka consumers?"],
}


@tool
def generate_interview_questions(candidate_id: str) -> str:
    """Create tailored screening questions for one candidate: depth probes on their strongest must-have skills,
    probes on gaps and years shortfalls, resume-specific impact questions, and behavioural questions."""
    reqs = CONTEXT["requirements"] or parse_requirements("")
    r = get_resume(candidate_id)
    b = _score_for(candidate_id)
    prof = deep_profile(r, reqs)
    q: list[tuple[str, str, str]] = []  # (section, question, listen for)

    for m in sorted(b["matched_must"], key=lambda x: -x["years"])[:3]:
        bank = QUESTION_BANK.get(m["skill"]) or [f"Describe the most complex thing you built with {m['skill']}. What trade-offs did you make?"]
        q.append(("Technical depth", bank[0], f"Concrete detail consistent with {fmt_years(m['years'])} yrs of {m['skill']}"))
        if len(bank) > 1 and m["years"] >= 3:
            q.append(("Technical depth", bank[1], "Measured approach: profiling/data before changes"))
    for s in b["missing_must"]:
        related = sorted(x for x in r.all_skills() if SKILL_TAXONOMY.get(x, ("",))[0] == SKILL_TAXONOMY.get(s, ("",))[0])
        hint = f" You've worked with {', '.join(related[:2])} — how would that transfer?" if related else ""
        q.append(("Gap probe", f"The role needs {s}, which isn't on your resume. Have you used it at all?{hint} "
                               f"How would you ramp up in your first 30 days?", "Honest self-assessment and a realistic plan"))
    for s in b["short_years"]:
        q.append(("Gap probe", f"You have about {fmt_years(s['have'])} year(s) of {s['skill']} against our {fmt_years(s['need'])}+. "
                               f"What's the deepest {s['skill']} problem you've owned end to end?", "Depth that compensates for fewer years"))
    for s in prof["claimed_not_demonstrated"]:
        q.append(("Verify claim", f"{s} is in your skills list but not in your role descriptions. Where have you used it in production?",
                  "A specific project, not coursework"))
    if prof["quantified"]:
        bullet = prof["quantified"][0]
        q.append(("Resume deep-dive", f"You wrote: \"{bullet[:120]}\". What was your personal contribution, and how was it measured?",
                  "Clear ownership vs team result; how the metric was measured"))
    elif prof["n_bullets"]:
        q.append(("Resume deep-dive", "Pick one project from your resume: what changed for users or the business because of your work?",
                  "Ability to articulate outcomes, not just tasks"))
    for s in b["missing_nice"][:1]:
        q.append(("Nice-to-have", f"Any exposure to {s}? How would you approach learning it on the job?", "Curiosity, learning approach"))
    senior = (reqs.get("min_years") or 0) >= 5 or r.total_years >= 6
    if senior:
        q.append(("Behavioural", "Tell me about a technical decision you drove that others disagreed with. How did you get alignment?",
                  "Influence without authority; data-driven persuasion"))
        q.append(("Behavioural", "How have you helped a less experienced engineer grow?", "Specific mentoring examples"))
    else:
        q.append(("Behavioural", "Tell me about a time you got stuck. How did you unblock yourself and when did you ask for help?",
                  "Self-sufficiency balanced with collaboration"))
    if prof["job_hopping"]:
        q.append(("Behavioural", "Your recent roles have been fairly short. What are you looking for in your next team?", "Motivation and fit"))
    if prof["career_gaps"]:
        a, c = prof["career_gaps"][0]
        q.append(("Behavioural", f"I see a break between {a} and {c}. Anything you'd like to share about that time?",
                  "Neutral question — not a red flag by itself"))
    q.append(("Role fit", f"What interests you about the {reqs.get('title', 'role')} role specifically?", "Genuine motivation"))

    lines = [f"### Screening questions for {r.name} ({candidate_id})",
             f"_Score {b['score']} · {one_line_reason(b, reqs)}_\n"]
    section = None
    for i, (sec, question, listen) in enumerate(q, 1):
        if sec != section:
            lines.append(f"\n**{sec}**")
            section = sec
        lines.append(f"{i}. {question}\n   - _Listen for:_ {listen}")
    return "\n".join(lines)


AGENT_TOOLS = FS_TOOLS + RAG_TOOLS + [extract_requirements, compare_candidates, generate_interview_questions]
TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}


# =============================================================================
# 3. Natural-language understanding (rule-based; optional LLM fallback)
# =============================================================================
NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("done", re.compile(r"^\s*(done|exit|quit|bye|goodbye|end|stop|finish|that'?s all|thanks,? that'?s all)\b[.!]*\s*$", re.I)),
    ("help", re.compile(r"^\s*(help|\?|what can you do|commands)\b", re.I)),
    ("reset", re.compile(r"\b(reset|start over|go back to (the )?(original|jd|job description)|original requirements|undo all)\b", re.I)),
    ("questions", re.compile(r"\b(interview|screening|phone[- ]screen) questions?\b|\bquestions? (for|to ask)\b|\bwhat should i ask\b", re.I)),
    ("compare", re.compile(r"\b(compare|comparison|side[- ]by[- ]side|head[- ]to[- ]head|versus|vs\.?)\b", re.I)),
    ("explain", re.compile(r"^\s*(why|how come|explain|what makes)\b|\brank(ed)? (higher|lower|above|below)\b|\branking(s)? change", re.I)),
    ("screen", re.compile(r"\b(multi[- ]?round|screening|screen (them|the|candidates)|hire|no[- ]hire|hiring recommendation|"
                          r"final round|second round|deep analysis|recommendations?)\b", re.I)),
    ("detail", re.compile(r"\b(tell me (more )?about|report (for|on)|details? (for|on|of|about)|profile (of|for)|strengths|gaps|"
                          r"deep[- ]dive|match report|more about|how good is|assess)\b", re.I)),
    ("requirements", re.compile(r"^\s*(show|what are|list|display|current)\b.*\b(requirements|criteria)\b", re.I)),
    ("shortlist", re.compile(r"^\s*(show|list|display|see|give)\b.*\b(shortlist|ranking|rankings|top\s*\d*|results|leaderboard)\b", re.I)),
]
SEARCH_RE = re.compile(r"^\s*(please\s+)?(can you\s+)?(find|search|look for|get|show|list|give|who are|who has|any)\b", re.I)
REFINE_RE = re.compile(r"\b(add|remove|drop|exclude|require[sd]?|requirement|must|mandatory|nice[- ]to[- ]have|optional|bonus|"
                       r"a plus|preferred|years?|yrs|prioriti[sz]e|emphasi[sz]e|weight|focus on|care more|ignore|instead|"
                       r"change|update|lower|raise|increase|decrease|relax|tighten|only|degree|bachelor|master|phd|location|"
                       r"based in|also need|actually|doesn'?t matter|not needed|don'?t need)\b", re.I)


def classify_intent(text: str) -> str:
    for name, pat in INTENT_PATTERNS:
        if pat.search(text):
            if name == "shortlist" and (user_skills(text) or YEARS_RE.search(text)):
                return "search"
            return name
    if SEARCH_RE.search(text) and (user_skills(text) or YEARS_RE.search(text)):
        return "search"
    mentions_location = any(re.search(rf"\b{re.escape(l)}\b", text, re.I) for l in _known_locations())
    if REFINE_RE.search(text) and (user_skills(text) or YEARS_RE.search(text) or mentions_location or
                                   re.search(r"experience|education|degree|location|weight|skills", text, re.I)):
        return "refine"
    if user_skills(text) and len(text.split()) <= 6:
        return "refine"  # e.g. "also AWS"
    return "other"


def resolve_candidates(text: str, state: AgentState) -> tuple[list[str], list[str]]:
    """Map 'top 3', '#2', 'cand_014', 'John', 'Jane Doe' to candidate ids (in mention order)."""
    ranking = state.get("ranking", [])
    resumes = load_all_resumes()
    hits: list[tuple[int, str]] = []
    notes: list[str] = []

    m = re.search(r"\btop\s*(\d+|" + "|".join(NUM_WORDS) + r")\b", text, re.I)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1).lower()]
        hits += [(m.start() + i / 100, c) for i, c in enumerate(ranking[:n])]
    for m in re.finditer(r"(?:#|\brank\s*|\bnumber\s*|\bno\.\s*)(\d+)\b|\b(\d+)(?:st|nd|rd|th)\b", text, re.I):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= len(ranking):
            hits.append((m.start(), ranking[n - 1]))
    for m in re.finditer(r"\b(?:cand_\d{3}|up_[a-z0-9_]+)\b", text):
        if m.group(0) in resumes:
            hits.append((m.start(), m.group(0)))

    rank_pos = {c: i for i, c in enumerate(ranking)}
    by_full, by_first, by_last = {}, {}, {}
    for cid, r in resumes.items():
        parts = r.name.split()
        by_full[r.name.lower()] = cid
        by_first.setdefault(parts[0].lower(), []).append(cid)
        if len(parts) > 1:
            by_last.setdefault(parts[-1].lower(), []).append(cid)
    taken = []
    for full, cid in by_full.items():
        for m in re.finditer(rf"\b{re.escape(full)}\b", text, re.I):
            hits.append((m.start(), cid)); taken.append((m.start(), m.end()))
    for m in re.finditer(r"\b[A-Za-z][a-z]+\b", text):
        if any(a <= m.start() < b for a, b in taken) or (m.group(0)[0].islower() and not ranking):
            continue
        word = m.group(0).lower()
        cands = by_first.get(word) if m.group(0)[0].isupper() or word in by_first else None
        cands = cands or (by_last.get(word) if m.group(0)[0].isupper() else None)
        if not cands:
            continue
        best = min(cands, key=lambda c: rank_pos.get(c, 10_000))
        if len(cands) > 1:
            notes.append(f"'{m.group(0)}' matches {len(cands)} candidates — using {resumes[best].name} "
                         f"(the highest ranked); say the full name to pick another.")
        hits.append((m.start(), best))
    hits.sort()
    return list(dict.fromkeys(c for _, c in hits)), notes


def strip_search_prefix(text: str) -> str:
    return re.sub(r"^\s*(please\s+)?(can you\s+)?(find|search( for)?|look for|get|show|list|give)( me)?( all)?( the)?\s*"
                  r"(candidates?|people|engineers?|developers?|devs|profiles|resumes|someone|anyone|applicants?)?\s*"
                  r"(who (have|has|know|knows)|with|having|that (have|has)|skilled in|experienced in|for)?\s*", "", text, flags=re.I)


WEIGHT_WORDS = {"experience": "experience", "seniority": "experience", "years": "experience", "must": "must_have",
                "core skills": "must_have", "skills": "must_have", "nice": "nice_to_have", "bonus": "nice_to_have",
                "education": "education", "degree": "education", "relevance": "relevance", "semantic": "relevance"}
REMOVE_MARK = re.compile(r"\b(remove|drop|exclude|forget|ignore|no longer|don'?t need|do not need|not (required|needed|necessary)|"
                         r"isn'?t (required|needed)|doesn'?t matter|without)\b", re.I)


def apply_refinement(reqs: dict, text: str) -> tuple[dict, list[str]]:
    """Apply a natural-language change to the requirements. Returns (new_requirements, human-readable changes)."""
    new = json.loads(json.dumps(reqs))
    new.setdefault("weights", dict(DEFAULT_WEIGHTS))
    changes: list[str] = []
    must = {m["skill"]: m.get("min_years") for m in new["must_have"]}
    nice = [n["skill"] for n in new["nice_to_have"]]

    clauses = re.split(r"[;.]\s*|,\s*(?:and\s+)?(?=(?:also|then|make|add|remove|drop|require|set|change|move|increase|"
                       r"decrease|lower|raise|prioriti[sz]e|ignore|only)\b)|\s+and\s+(?=(?:also\s+)?(?:make|add|remove|drop|require|"
                       r"set|change|move|increase|decrease|lower|raise|prioriti[sz]e|ignore)\b)", text, flags=re.I)
    for clause in (c.strip() for c in clauses if c and c.strip()):
        skills = user_skills(clause)
        low = clause.lower()

        # location
        if re.search(r"\b(any|remove|drop|ignore)\b.*\blocation\b|\blocation\b.*\b(doesn'?t matter|any)\b", low):
            if new.get("location"):
                new["location"] = None; changes.append("removed the location filter")
            continue
        loc_hit = next((l for l in _known_locations() if re.search(rf"\b(in|from|based in|located in|near|only)\s+{re.escape(l)}\b", clause, re.I)
                        or (l.lower() == "remote" and re.search(r"\bremote\b", low))), None)
        if loc_hit and not skills:
            locs = list(new.get("location") or [])
            if re.search(r"\b(or|also)\b", low) and locs:
                locs.append(loc_hit)
            else:
                locs = [loc_hit]
            new["location"] = list(dict.fromkeys(locs)); changes.append(f"location filter → {', '.join(new['location'])}")
            continue

        # education
        if re.search(r"\b(degree|bachelor|master|ph\.?d|education)\b", low) and not skills:
            if REMOVE_MARK.search(low) or "optional" in low:
                if new.get("education"):
                    new["education"] = None; changes.append("dropped the education requirement")
                new["weights"]["education"] = 0.0
            else:
                lvl = "PhD" if re.search(r"ph\.?d", low) else "Masters" if "master" in low else "Bachelors"
                new["education"] = {"level": lvl, "required": not NICE_MARK.search(low)}
                new["weights"]["education"] = new["weights"].get("education") or DEFAULT_WEIGHTS["education"]
                changes.append(f"education → {lvl} ({'required' if new['education']['required'] else 'preferred'})")
            continue

        # weights
        wm = re.search(r"\b(prioriti[sz]e|emphasi[sz]e|focus (more )?on|care (more )?about|weight|more weight|less weight|"
                       r"de-?prioriti[sz]e|care less)\b", low)
        if wm and not skills:
            comp = next((v for k, v in WEIGHT_WORDS.items() if k in low), None)
            if comp:
                down = bool(re.search(r"less|de-?prioriti|lower|reduce", low))
                old = new["weights"].get(comp, 0)
                new["weights"][comp] = round(max(0.0, old * 0.5) if down else min(0.6, old * 2 if old else 0.15), 3)
                changes.append(f"{'decreased' if down else 'increased'} weight on {COMPONENT_LABELS[comp].lower()} "
                               f"({old:.0%} → {new['weights'][comp]:.0%})")
                continue

        if skills:
            years = {}
            for m in YEARS_RE.finditer(clause):
                after = [s for s in user_skills(clause[m.end(): m.end() + 30]) if s in skills]
                before = [s for s in user_skills(clause[max(0, m.start() - 30): m.start()]) if s in skills]
                target = (after or before or [None])[0]
                if target:
                    years[target] = float(m.group(1))
            for s in skills:
                if REMOVE_MARK.search(clause):
                    if s in must:
                        must.pop(s); changes.append(f"removed {s} from must-haves")
                    if s in nice:
                        nice.remove(s); changes.append(f"removed {s} from nice-to-haves")
                elif NICE_MARK.search(clause) or re.search(r"\bnot (a )?must\b|\bdowngrade\b", low):
                    if s in must:
                        must.pop(s)
                    if s not in nice:
                        nice.append(s)
                    changes.append(f"{s} → nice-to-have")
                else:
                    prev = must.get(s, "absent") if s in must else "absent"
                    y = years.get(s, must.get(s))
                    must[s] = y
                    if s in nice:
                        nice.remove(s)
                    if prev == "absent":
                        changes.append(f"{s} → must-have" + (f" ({fmt_years(y)}+ yrs)" if y else ""))
                    elif y != prev:
                        changes.append(f"{s} minimum → {fmt_years(y)}+ yrs")
            continue

        ym = YEARS_RE.search(clause)
        if ym:
            new["min_years"] = float(ym.group(1)); changes.append(f"overall experience → {fmt_years(new['min_years'])}+ yrs")
        elif re.search(r"\bexperience\b", low):
            if REMOVE_MARK.search(low):
                new["min_years"] = None; changes.append("removed the minimum-experience requirement")
            elif re.search(r"\b(relax|lower|reduce|loosen)\b", low) and new.get("min_years"):
                new["min_years"] = max(0.0, new["min_years"] - 2); changes.append(f"relaxed experience → {fmt_years(new['min_years'])}+ yrs")
            elif re.search(r"\b(raise|increase|tighten|more)\b", low):
                new["min_years"] = (new.get("min_years") or 0) + 2; changes.append(f"raised experience → {fmt_years(new['min_years'])}+ yrs")

    new["must_have"] = [{"skill": s, "min_years": y} for s, y in must.items()]
    new["nice_to_have"] = [{"skill": s} for s in nice if s not in must]
    return new, changes


# Optional LLM ---------------------------------------------------------------------------
_LLM = None


def get_llm():
    global _LLM
    spec = os.environ.get("MATCHER_LLM")
    if _LLM is None and spec:
        try:
            from langchain.chat_models import init_chat_model
            _LLM = init_chat_model(spec, temperature=0)
        except Exception as e:  # pragma: no cover - depends on local setup
            print(f"[matching_agent] LLM disabled ({e})", file=sys.stderr)
            _LLM = False
    return _LLM or None


HELP_TEXT = """Here's what I can do:
- **Search:** "Find me candidates with React and 3+ years experience"
- **Refine:** "Make AWS a must-have", "Drop the REST requirement", "Require 7+ years", "Prioritize experience", "Only candidates in Chennai"
- **Compare:** "Compare the top 3 matches side by side", "Compare John and Jane"
- **Explain:** "Why did John rank higher than Jane?", "Why is #4 ranked there?", "Explain the ranking changes"
- **Reports:** "Tell me about Jane Doe" (strengths, gaps, improvement suggestions)
- **Interview prep:** "Interview questions for #1"
- **Screening:** "Run multi-round screening" (top 10 → deep analysis → hire/no-hire)
- **Status:** "Show the shortlist", "Show requirements", "Reset to the original JD", "done\""""


# =============================================================================
# 4. Graph nodes
# =============================================================================
def parse_jd(state: AgentState) -> dict:
    jd = (state.get("jd_text") or "").strip()
    if jd:
        jd = re.sub(r"\r\n?", "\n", jd)
        jd = re.sub(r"\n{3,}", "\n\n", jd)
        return {"jd_text": jd, "requirements_source": jd, "requirements_mode": "jd",
                "reasoning_log": [f"parse_jd: loaded JD ({len(jd.split())} words)"]}
    query = next((m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), "")
    return {"requirements_source": strip_search_prefix(query), "requirements_mode": "query",
            "reasoning_log": [f"parse_jd: no JD supplied; using query '{query}'"]}


def extract_requirements_node(state: AgentState) -> dict:
    src = state["requirements_source"]
    reqs = extract_requirements.invoke({"jd": src})
    if state.get("requirements_mode") == "query":
        reqs["title"] = f"Search: {src[:60]}"
    upd: dict[str, Any] = {"requirements": reqs,
                           "requirement_history": [{"version": len(state.get("requirement_history", [])) + 1,
                                                    "source": state.get("requirements_mode"), "requirements": reqs}],
                           "reasoning_log": [f"extract_requirements: {requirements_summary(reqs)}"]}
    if not state.get("original_requirements"):
        upd["original_requirements"] = reqs
    if state.get("requirements"):  # a new search replaces the previous criteria
        upd["previous_requirements"] = state["requirements"]
        upd["change_note"] = f"new search: {src}"
    if not reqs["must_have"] and not reqs["nice_to_have"] and not reqs.get("min_years"):
        upd["messages"] = [AIMessage("I couldn't identify concrete skills or experience levels in that, so I'll rank on "
                                     "overall resume relevance only. Try naming skills, e.g. 'React, Node.js, 3+ years'.")]
    return upd


def search_resumes(state: AgentState) -> dict:
    reqs = state["requirements"]
    query = " ".join([reqs.get("title", "")] + [m["skill"] for m in reqs["must_have"]] * 2 +
                     [n["skill"] for n in reqs["nice_to_have"]])
    hits = semantic_search(query, k=SEARCH_K)
    return {"candidate_pool": [{"candidate_id": h["candidate_id"], "name": h["name"], "relevance": h["score"],
                                "snippet": h["snippet"]} for h in hits],
            "reasoning_log": [f"search_resumes: RAG query '{query.strip()}' → {len(hits)} candidates"]}


def rank_candidates(state: AgentState) -> dict:
    reqs = state["requirements"]
    pool = state["candidate_pool"]
    relevance = {p["candidate_id"]: p["relevance"] for p in pool}
    resumes = load_all_resumes()
    # every resume is scored (round-1 screen); RAG relevance is one explainable component
    scores = {cid: score_candidate(r, reqs, relevance.get(cid, 0.0)) for cid, r in resumes.items()}
    ranking = sorted((c for c, b in scores.items() if not b["excluded"]),
                     key=lambda c: (-scores[c]["score"], -scores[c]["total_years"]))
    shortlist = [{"rank": i + 1, "candidate_id": c, "name": scores[c]["name"], "score": scores[c]["score"],
                  "reasoning": one_line_reason(scores[c], reqs)} for i, c in enumerate(ranking[:TOP_K])]

    upd: dict[str, Any] = {"all_scores": scores, "ranking": ranking, "shortlist": shortlist, "rank_changes": [],
                           "screening": {},
                           "reasoning_log": [f"rank_candidates: scored {len(scores)}; top = "
                                             + ", ".join(f"{s['name']} {s['score']}" for s in shortlist[:3])]}
    prev_reqs = state.get("previous_requirements")
    if state.get("all_scores") and prev_reqs and state.get("change_note"):
        upd["previous_scores"] = state["all_scores"]
        upd["previous_ranking"] = state.get("ranking", [])
        upd["rank_changes"] = explain_rank_changes(state["all_scores"], scores, state.get("ranking", []), ranking,
                                                   prev_reqs, reqs, TOP_K)
    return upd


def _shortlist_table(state: AgentState, n: int = TOP_K) -> str:
    reqs, scores = state["requirements"], state["all_scores"]
    lines = ["| # | Candidate | Score | Must-haves | Exp | Notes |", "|---|---|---|---|---|---|"]
    for s in state["shortlist"][:n]:
        b = scores[s["candidate_id"]]
        note = ("missing " + ", ".join(b["missing_must"])) if b["missing_must"] else \
               (", ".join(f"{x['skill']} {fmt_years(x['have'])}/{fmt_years(x['need'])}y" for x in b["short_years"]) or
                ("+ " + ", ".join(b["matched_nice"][:3]) if b["matched_nice"] else ""))
        lines.append(f"| {s['rank']} | {s['name']} (`{s['candidate_id']}`) | {s['score']} | "
                     f"{len(b['matched_must'])}/{len(reqs['must_have'])} | {fmt_years(b['total_years'])}y | {note} |")
    return "\n".join(lines)


def _changes_md(changes: list[dict], limit: int = 8) -> str:
    out = []
    for c in changes[:limit]:
        if c["entered_top"]:
            head = f"⬆ **{c['name']}** entered the top {TOP_K} (#{c['old_rank'] or '–'} → #{c['new_rank']})"
        elif c["left_top"]:
            head = f"⬇ **{c['name']}** dropped out (#{c['old_rank']} → #{c['new_rank'] or 'filtered'})"
        else:
            arrow = "⬆" if (c["delta"] or 0) > 0 else "⬇"
            head = f"{arrow} **{c['name']}** #{c['old_rank']} → #{c['new_rank']}"
        out.append(f"- {head}: {c['reason']}")
    left = [c for c in changes if c["left_top"]]
    if left and not any(c["left_top"] for c in changes[:limit]):
        out.append("- Dropped out: " + ", ".join(f"{c['name']} ({c['reason']})" for c in left[:3]))
    return "\n".join(out)


def generate_report(state: AgentState) -> dict:
    _sync_context(state)
    reqs, scores, shortlist = state["requirements"], state["all_scores"], state["shortlist"]
    n_screened = len(scores)
    changes = state.get("rank_changes") or []

    md = [f"# Match report — {reqs['title']}", "", f"Screened **{n_screened}** resumes · generated {timestamp()}", "",
          "## Requirements", requirements_summary(reqs, multiline=True), ""]
    if state.get("change_note") and changes:
        md += ["## What changed", f"_Change: {state['change_note']}_", "", _changes_md(changes, 20), ""]
    md += ["## Shortlist", _shortlist_table(state), "", "## Candidate details"]
    for s in shortlist:
        r = get_resume(s["candidate_id"])
        b = scores[s["candidate_id"]]
        st, gp = strengths_and_gaps(r, b, reqs)
        md += [f"### {s['rank']}. {s['name']} — {s['score']}", "_" + " · ".join(x for x in [r.title, r.location, one_line_reason(b, reqs)] if x) + "_", "",
               "**Strengths**", *[f"- {x}" for x in st or ["—"]], "", "**Gaps**", *[f"- {x}" for x in gp or ["None significant"]], ""]
    path = save_report(f"shortlist_{timestamp()}.md", "\n".join(md))

    chat = []
    if state.get("change_note") and changes:
        chat += [f"Re-ranked after: _{state['change_note']}_\n", "**Ranking changes**", _changes_md(changes), ""]
    elif state.get("change_note"):
        chat += [f"Re-ranked after: _{state['change_note']}_ — the top {TOP_K} order didn't change.\n"]
    chat += [f"**Top {len(shortlist)} for {reqs['title']}** (screened {n_screened} resumes)",
             requirements_summary(reqs), "", _shortlist_table(state), "",
             f"Full report with strengths and gaps: `{path}`",
             "\nTry: _\"Compare the top 3\"_, _\"Why did #1 rank above #2?\"_, _\"Make AWS a must-have\"_, "
             "_\"Run multi-round screening\"_, or _\"done\"_."]
    return {"messages": [AIMessage("\n".join(chat))], "last_report_path": path, "change_note": "",
            "reasoning_log": [f"generate_report: wrote {path}"]}


ROUTES = {"search": "extract_requirements", "refine": "refine_requirements", "compare": "compare_node",
          "explain": "explain_node", "questions": "interview_node", "detail": "candidate_report_node",
          "screen": "screening_node", "shortlist": "status_node", "requirements": "status_node", "help": "status_node",
          "other": "tool_agent"}


def human_feedback(state: AgentState) -> Command:
    """Human-in-the-loop: pause, wait for the user's next message, route on its intent."""
    user = interrupt({"prompt": "What would you like to do next? (type 'help' for options, 'done' to finish)"})
    text = str(user).strip()
    intent = classify_intent(text) if text else "help"
    upd: dict[str, Any] = {"messages": [HumanMessage(text)], "last_user_message": text, "intent": intent,
                           "reasoning_log": [f"human_feedback: '{text}' → {intent}"]}
    if intent == "done":
        upd["messages"].append(AIMessage(f"Done. Final shortlist saved at `{state.get('last_report_path', 'reports/')}`. Good luck with hiring!"))
        return Command(update=upd, goto=END)
    if intent == "reset":
        orig = state.get("original_requirements")
        upd.update({"previous_requirements": state["requirements"], "requirements": orig,
                    "change_note": "reset to the original job requirements"})
        return Command(update=upd, goto="search_resumes")
    if intent == "search":
        upd.update({"requirements_source": strip_search_prefix(text), "requirements_mode": "query"})
    return Command(update=upd, goto=ROUTES[intent])


def refine_requirements(state: AgentState) -> Command:
    text = state["last_user_message"]
    new, changes = apply_refinement(state["requirements"], text)
    if not changes:
        msg = ("I couldn't turn that into a requirement change. Examples: 'make AWS a must-have', "
               "'Docker is optional', 'require 7+ years', 'drop GraphQL', 'prioritize experience'.\n\n"
               f"Current requirements: {requirements_summary(state['requirements'])}")
        return Command(update={"messages": [AIMessage(msg)]}, goto="human_feedback")
    note = "; ".join(changes)
    return Command(update={"previous_requirements": state["requirements"], "requirements": new, "change_note": note,
                           "requirement_history": [{"version": len(state.get("requirement_history", [])) + 1,
                                                    "source": "refinement", "change": note, "requirements": new}],
                           "reasoning_log": [f"refine_requirements: {note}"]},
                   goto="search_resumes")


def compare_node(state: AgentState) -> dict:
    _sync_context(state)
    ids, notes = resolve_candidates(state["last_user_message"], state)
    if len(ids) < 2:
        ids = state["ranking"][:3]
        notes.append("No specific candidates recognised — comparing the top 3.")
    table = compare_candidates.invoke({"candidate_ids": ids})
    return {"messages": [AIMessage("\n".join(notes + ["", table]) if notes else table)],
            "reasoning_log": [f"compare: {ids}"]}


def _what_if(cid: str, state: AgentState) -> float | None:
    b = state["all_scores"][cid]
    if not b["missing_must"]:
        return None
    rel = b["relevance"]
    return score_candidate(get_resume(cid), state["requirements"], rel, assume_skills=frozenset(b["missing_must"]))["score"]


def explain_node(state: AgentState) -> dict:
    _sync_context(state)
    text = state["last_user_message"]
    ids, notes = resolve_candidates(text, state)
    ranking, scores, reqs = state["ranking"], state["all_scores"], state["requirements"]
    rank = lambda c: ranking.index(c) + 1 if c in ranking else None

    if not ids and re.search(r"chang|move|re-?rank|shift", text, re.I):
        ch = state.get("rank_changes") or []
        body = _changes_md(ch, 12) if ch else "There haven't been any requirement changes yet, so the ranking hasn't moved."
        return {"messages": [AIMessage("**Why the ranking changed**\n" + body)]}
    if not ids:
        ids = ranking[:2]
        notes.append("No candidate named — explaining #1 vs #2.")
    if len(ids) == 1:
        c = ids[0]
        r = rank(c)
        if r is None:
            body = f"{scores[c]['name']} is filtered out by the current location filter."
        elif r == 1:
            body = f"**{scores[c]['name']}** is #1.\n\n" + explain_pair(scores[c], scores[ranking[1]], get_resume(c),
                                                                        get_resume(ranking[1]), reqs, 1, 2)
        else:
            above = ranking[r - 2]
            body = (f"**{scores[c]['name']}** is #{r} with {scores[c]['score']} ({one_line_reason(scores[c], reqs)}). "
                    f"Here's what separates them from #{r - 1}:\n\n" +
                    explain_pair(scores[above], scores[c], get_resume(above), get_resume(c), reqs, r - 1, r,
                                 what_if_score=_what_if(c, state)))
    else:
        a, b = ids[0], ids[1]
        hi, lo = (a, b) if scores[a]["score"] >= scores[b]["score"] else (b, a)
        body = explain_pair(scores[hi], scores[lo], get_resume(hi), get_resume(lo), reqs, rank(hi), rank(lo),
                            what_if_score=_what_if(lo, state))
        asked_hi = re.search(rf"{re.escape(scores[a]['name'].split()[0])}\b.*\b(higher|above|better|over)", text, re.I)
        if asked_hi and hi != a:
            body = f"Actually, {scores[b]['name']} currently ranks higher than {scores[a]['name']}.\n\n" + body
    return {"messages": [AIMessage("\n".join(notes + [body]) if notes else body)],
            "reasoning_log": [f"explain: {ids[:2]}"]}


def interview_node(state: AgentState) -> dict:
    _sync_context(state)
    ids, notes = resolve_candidates(state["last_user_message"], state)
    if not ids:
        ids = state["ranking"][:1]
        notes.append("No candidate named — generating questions for #1.")
    out = [generate_interview_questions.invoke({"candidate_id": c}) for c in ids[:3]]
    return {"messages": [AIMessage("\n\n".join(notes + out))], "reasoning_log": [f"interview_questions: {ids[:3]}"]}


def build_match_report(cid: str, state: AgentState, verdict: dict | None = None) -> str:
    reqs, b = state["requirements"], state["all_scores"][cid]
    r = get_resume(cid)
    prof = deep_profile(r, reqs)
    st, gp = strengths_and_gaps(r, b, reqs, prof)
    rank = state["ranking"].index(cid) + 1 if cid in state["ranking"] else None
    md = [f"## Match report: {r.name} (`{cid}`)",
          " · ".join(x for x in [f"**{r.title}**" if r.title else "", r.location, f"{fmt_years(r.total_years)} yrs", b["degree"],
                                 f"rank #{rank or '–'} of {len(state['ranking'])}"] if x),
          f"\n**Overall score: {b['score']}/100**\n", "| Component | Raw | Points |", "|---|---|---|"]
    for k, v in b["contributions"].items():
        md.append(f"| {COMPONENT_LABELS[k]} | {b['components'][k]:.2f} | {v:.1f} |")
    if b["penalty"]:
        md.append(f"| Missing must-have penalty | | −{b['penalty']:.1f} |")
    md += ["\n**Must-have evidence**", "| Skill | Listed | Years in roles | Used currently | Status |", "|---|---|---|---|---|"]
    for e in prof["evidence"]:
        ok = "✓" if e["skill"] not in b["missing_must"] and not any(x["skill"] == e["skill"] for x in b["short_years"]) else \
             ("⚠ short" if e["skill"] not in b["missing_must"] else "✗ missing")
        md.append(f"| {e['skill']}{' (' + fmt_years(e['need']) + '+)' if e['need'] else ''} | {'yes' if e['listed'] else 'no'} | "
                  f"{fmt_years(e['years'])} | {'yes' if e['recent'] else 'no'} | {ok} |")
    md += ["\n**Strengths**", *[f"- {x}" for x in st or ["—"]], "\n**Gaps**", *[f"- {x}" for x in gp or ["None significant"]]]
    borderline = verdict and verdict["verdict"].startswith("Borderline")
    near = b["score"] >= 50 and (b["missing_must"] or b["short_years"] or (reqs.get("min_years") or 0) > r.total_years)
    if borderline or near or verdict is None:
        md += ["\n**Improvement suggestions**" + (" (borderline candidate)" if borderline or near else ""),
               *[f"- {x}" for x in improvement_suggestions(r, b, reqs, prof)]]
    if verdict:
        md.append(f"\n**Recommendation:** {verdict['verdict']} (confidence {verdict['confidence']}, final score {verdict['final_score']})")
    return "\n".join(md)


def candidate_report_node(state: AgentState) -> dict:
    _sync_context(state)
    ids, notes = resolve_candidates(state["last_user_message"], state)
    if not ids:
        ids = state["ranking"][:1]
        notes.append("No candidate named — showing the report for #1.")
    verdicts = (state.get("screening") or {}).get("decisions", {})
    reports = [build_match_report(c, state, verdicts.get(c)) for c in ids[:3]]
    path = save_report(f"match_report_{'_'.join(ids[:3])}_{timestamp()}.md", "\n\n".join(reports))
    return {"messages": [AIMessage("\n".join(notes) + ("\n\n" if notes else "") + "\n\n".join(reports) + f"\n\n_Saved to `{path}`_")],
            "last_report_path": path, "reasoning_log": [f"candidate_report: {ids[:3]}"]}


def screening_node(state: AgentState) -> dict:
    """Multi-round screening: R1 top-10 of all resumes → R2 deep evidence analysis → R3 hire / no-hire."""
    _sync_context(state)
    reqs, scores, ranking = state["requirements"], state["all_scores"], state["ranking"]
    n = len(scores)
    r1 = ranking[:FINAL_K]
    cutoff = scores[r1[-1]]["score"] if r1 else 0

    deep, decisions = {}, {}
    for c in r1:
        prof = deep_profile(get_resume(c), reqs)
        d = deep_score(scores[c], prof, reqs)
        deep[c] = {**d, "profile": prof}
        decisions[c] = hire_decision(scores[c], d, reqs)
    tier = {"Strong Hire": 0, "Hire": 1, "Borderline (Lean No Hire)": 2, "No Hire": 3}
    final_order = sorted(r1, key=lambda c: (tier[decisions[c]["verdict"]], -deep[c]["final_score"]))

    md = [f"# Multi-round screening — {reqs['title']}", "", requirements_summary(reqs, multiline=True), "",
          f"## Round 1 — initial screen: {len(r1)} of {n} resumes advance (cut-off score {cutoff})",
          "| R1 rank | Candidate | R1 score | Summary |", "|---|---|---|---|"]
    md += [f"| {i} | {scores[c]['name']} | {scores[c]['score']} | {one_line_reason(scores[c], reqs)} |" for i, c in enumerate(r1, 1)]
    md += ["", "## Round 2 — deep analysis",
           "| Candidate | Evidence depth | Seniority | Impact | Leadership | Stability | Certs/extras | Evidence score | Final |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in final_order:
        p = deep[c]["evidence_parts"]
        md.append(f"| {scores[c]['name']} | {p['evidence_depth']:.2f} | {p['seniority']:.2f} | {p['impact']:.2f} | {p['leadership']:.2f} | "
                  f"{p['stability']:.2f} | {p['certs_and_extras']:.2f} | {deep[c]['evidence_score']} | {deep[c]['final_score']} |")
    md += ["", "_Final = 50% round-1 match + 50% evidence score (skill years shown in actual roles, recency, seniority vs the bar, quantified impact, "
           "leadership, tenure stability, relevant certifications)._", "",
           "## Round 3 — recommendations", "| Final rank | Candidate | Verdict | Confidence | Final score | Rationale |",
           "|---|---|---|---|---|---|"]
    for i, c in enumerate(final_order, 1):
        st, gp = strengths_and_gaps(get_resume(c), scores[c], reqs, deep[c]["profile"])
        why = (st[0] if st else "") + ((" · gap: " + gp[0]) if gp else "")
        md.append(f"| {i} | {scores[c]['name']} | **{decisions[c]['verdict']}** | {decisions[c]['confidence']} | "
                  f"{deep[c]['final_score']} | {why} |")
    md += ["", "## Detailed match reports"]
    sub_state = {**state, "screening": {"decisions": decisions}}
    md += [build_match_report(c, sub_state, decisions[c]) for c in final_order]
    path = save_report(f"screening_{timestamp()}.md", "\n".join(md))

    counts: dict[str, int] = {}
    for d in decisions.values():
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
    moved = [(scores[c]["name"], r1.index(c) + 1, i + 1) for i, c in enumerate(final_order) if r1.index(c) != i]
    chat = [f"**Multi-round screening complete** ({n} resumes → {len(r1)} → decisions)\n",
            f"**Round 1:** top {len(r1)} advance (cut-off {cutoff}).",
            "**Round 2:** deep evidence analysis" + (" re-ordered: " + ", ".join(f"{nm} #{a}→#{b}" for nm, a, b in moved[:5])
                                                    if moved else " confirmed the round-1 order."),
            "**Round 3:** " + ", ".join(f"{v} × {k}" for k, v in counts.items()), "",
            "| # | Candidate | Verdict | Final | Confidence |", "|---|---|---|---|---|"]
    chat += [f"| {i} | {scores[c]['name']} | {decisions[c]['verdict']} | {deep[c]['final_score']} | {decisions[c]['confidence']} |"
             for i, c in enumerate(final_order, 1)]
    border = [c for c in final_order if decisions[c]["verdict"].startswith("Borderline")]
    if border:
        chat += ["", "**Borderline candidates — what would move them to Hire:**"]
        for c in border[:3]:
            sug = improvement_suggestions(get_resume(c), scores[c], reqs, deep[c]["profile"])
            chat.append(f"- **{scores[c]['name']}:** " + " ".join(sug[:2]))
    chat += ["", f"Full 3-round report with per-candidate strengths, gaps and suggestions: `{path}`"]
    screening = {"round1": r1, "cutoff": cutoff,
                 "round2": {c: {k: v for k, v in deep[c].items() if k != "profile"} for c in r1},
                 "final_order": final_order, "decisions": decisions, "report_path": path}
    return {"messages": [AIMessage("\n".join(chat))], "screening": screening, "last_report_path": path,
            "reasoning_log": [f"screening: {counts}"]}


def status_node(state: AgentState) -> dict:
    intent = state.get("intent")
    if intent == "help":
        return {"messages": [AIMessage(HELP_TEXT)]}
    if intent == "requirements":
        hist = state.get("requirement_history", [])
        changes = [h.get("change") for h in hist if h.get("change")]
        msg = f"**Current requirements — {state['requirements']['title']}**\n" + requirements_summary(state["requirements"], multiline=True)
        if changes:
            msg += "\n\n**Changes this session:**\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(changes, 1))
        return {"messages": [AIMessage(msg)]}
    m = re.search(r"\btop\s*(\d+)", state["last_user_message"], re.I)
    n = min(int(m.group(1)), TOP_K) if m else TOP_K
    return {"messages": [AIMessage(f"**Current shortlist** — {requirements_summary(state['requirements'])}\n\n" + _shortlist_table(state, n))]}


def tool_agent(state: AgentState) -> dict:
    """Free-form questions. With an LLM configured, it reasons and calls tools; otherwise it falls back to RAG search."""
    _sync_context(state)
    text = state["last_user_message"]
    llm = get_llm()
    if llm:
        sys_msg = SystemMessage(
            "You are a recruiting assistant inside a resume-matching agent. Use the tools to look things up; never invent "
            "candidate facts. Current requirements: " + requirements_summary(state["requirements"]) +
            "\nCurrent shortlist: " + "; ".join(f"#{s['rank']} {s['name']} ({s['candidate_id']}) {s['score']}" for s in state["shortlist"]))
        msgs: list[AnyMessage] = [sys_msg, HumanMessage(text)]
        bound = llm.bind_tools(AGENT_TOOLS)
        for _ in range(6):
            ai = bound.invoke(msgs)
            msgs.append(ai)
            if not getattr(ai, "tool_calls", None):
                return {"messages": [AIMessage(ai.content if isinstance(ai.content, str) else str(ai.content))],
                        "reasoning_log": [f"tool_agent: answered with {len(msgs) - 2} steps"]}
            for call in ai.tool_calls:
                try:
                    result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
                except Exception as e:  # surface tool errors to the model
                    result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result)[:6000], tool_call_id=call["id"]))
        return {"messages": [AIMessage("I ran out of steps answering that — could you narrow the question?")]}

    hits = semantic_search(text, k=5)
    if hits and hits[0]["score"] > 0 and user_skills(text):
        lines = [f"I'm not sure what you'd like me to change, so here are the resumes most relevant to “{text}”:"]
        lines += [f"- **{h['name']}** (`{h['candidate_id']}`) — {h['snippet']}" for h in hits]
        lines.append("\nSay _\"Find me candidates with …\"_ to make this the new search, or type _help_.")
        return {"messages": [AIMessage("\n".join(lines))]}
    return {"messages": [AIMessage("I didn't catch that.\n\n" + HELP_TEXT)]}


# =============================================================================
# 5. Graph assembly
# =============================================================================
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("parse_jd", parse_jd)
    g.add_node("extract_requirements", extract_requirements_node)
    g.add_node("search_resumes", search_resumes)
    g.add_node("rank_candidates", rank_candidates)
    g.add_node("generate_report", generate_report)
    g.add_node("human_feedback", human_feedback, destinations=tuple(set(ROUTES.values())) + ("search_resumes", END))
    g.add_node("refine_requirements", refine_requirements, destinations=("search_resumes", "human_feedback"))
    for name, fn in [("compare_node", compare_node), ("explain_node", explain_node), ("interview_node", interview_node),
                     ("candidate_report_node", candidate_report_node), ("screening_node", screening_node),
                     ("status_node", status_node), ("tool_agent", tool_agent)]:
        g.add_node(name, fn)
        g.add_edge(name, "human_feedback")

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes", "rank_candidates")
    g.add_edge("rank_candidates", "generate_report")
    g.add_edge("generate_report", "human_feedback")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class MatchingAgent:
    """Thin conversational wrapper: start() runs the pipeline, chat() resumes the human-feedback interrupt."""

    def __init__(self, thread_id: str | None = None, checkpointer=None):
        self.graph = build_graph(checkpointer)
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 50}
        get_retriever(rebuild=True)

    def _run(self, payload) -> str:
        before = len(self.state.get("messages", [])) if self._started else 0
        self.graph.invoke(payload, self.config)
        msgs = self.state.get("messages", [])[before:]
        return "\n\n".join(m.content for m in msgs if isinstance(m, AIMessage))

    _started = False

    def start(self, jd_text: str | None = None, jd_path: str | None = None, query: str | None = None) -> str:
        if jd_path:
            jd_text = read_text_file(jd_path)
        if not jd_text and not query:
            raise ValueError("Provide a job description (jd_text / jd_path) or a natural-language query.")
        init: dict[str, Any] = {"jd_text": jd_text or "", "messages": [HumanMessage(query or "Match candidates to this job description.")]}
        out = self._run(init)
        self._started = True
        return out

    def chat(self, message: str) -> str:
        if not self._started:
            return self.start(query=message)
        if self.finished:
            return "This session has ended. Start a new MatchingAgent to run another search."
        return self._run(Command(resume=message))

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.config).values

    @property
    def finished(self) -> bool:
        return self._started and not self.graph.get_state(self.config).next


# =============================================================================
# 6. CLI
# =============================================================================
DEMO_SCRIPT = [
    "Compare the top 3 matches side by side",
    "Why did John rank higher than Jane?",
    "Make AWS a must-have",
    "Why did John rank higher than Jane?",
    "Run multi-round screening",
    "Tell me about Jane Doe",
    "Interview questions for #1",
    "Find me candidates with React and 3+ years experience",
    "Reset to the original JD",
    "Prioritize experience and drop REST APIs",
    "Show requirements",
    "done",
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LangGraph resume matching agent")
    ap.add_argument("--jd", help="path to a job description text file")
    ap.add_argument("--query", help="natural-language search instead of a JD")
    ap.add_argument("--resumes", help="folder of resumes: .pdf, .docx, .txt or .md (default data/resumes)")
    ap.add_argument("--demo", action="store_true", help="run a scripted conversation showcasing every feature")
    ap.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = ap.parse_args(argv)

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        return
    if args.resumes:  # a folder of .pdf / .docx / .txt resumes; use only that folder
        fs_tools.set_resume_dir(args.resumes)
        fs_tools.configure_sources(include_samples=True, include_uploads=False)
    if not load_all_resumes():
        sys.exit(f"No readable resumes found in {fs_tools.RESUME_DIR}. Run `python generate_sample_data.py` first."
                 + "".join(f"\n  skipped {n}: {w}" for n, w in fs_tools.LOAD_WARNINGS.items()))
    for n, w in fs_tools.LOAD_WARNINGS.items():
        print(f"[skipped {n}: {w}]")

    agent = MatchingAgent()
    jd = args.jd or (None if args.query else "data/job_descriptions/senior_fullstack.txt")
    print(agent.start(jd_path=jd, query=args.query))
    script = iter(DEMO_SCRIPT) if args.demo else None
    while not agent.finished:
        if script:
            msg = next(script, "done")
            print(f"\n\033[1;36mYou:\033[0m {msg}")
        else:
            try:
                msg = input("\n\033[1;36mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                msg = "done"
        print("\n" + agent.chat(msg))


if __name__ == "__main__":
    main()
