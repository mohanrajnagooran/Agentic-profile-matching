"""
scoring.py — the explainable matching engine behind the agent.

Everything here is deterministic and transparent: every score is a weighted sum of
named components, so the agent can always say *why* a candidate ranks where they do.

    requirements  -> parse_requirements(text)
    initial screen -> score_candidate(resume, reqs, relevance)
    deep analysis  -> deep_profile(resume, reqs) + deep_score(...)
    final round    -> hire_decision(...)
    explainability -> strengths_and_gaps / improvement_suggestions / explain_pair / explain_rank_changes
"""
from __future__ import annotations

import re
from typing import Any

from skills import Resume, extract_skills, skill_category, SKILL_TAXONOMY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"must_have": 0.45, "experience": 0.20, "nice_to_have": 0.15, "relevance": 0.12, "education": 0.08}
COMPONENT_LABELS = {"must_have": "Must-have skills", "experience": "Experience", "nice_to_have": "Nice-to-have skills",
                    "relevance": "Resume relevance (RAG)", "education": "Education"}
MISSING_MUST_PENALTY = 0.85  # total score multiplied by this for every missing must-have
DEGREE_LEVEL = {"None listed": 0, "Other": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}

# Hire / no-hire thresholds on the final (round-3) score
STRONG_HIRE, HIRE, BORDERLINE = 82, 72, 60

CERT_FOR_SKILL = {"AWS": "AWS Certified Solutions Architect – Associate", "Kubernetes": "CKA (Certified Kubernetes Administrator)",
                  "Azure": "Azure Developer Associate", "GCP": "Google Cloud Professional Developer",
                  "Terraform": "HashiCorp Terraform Associate", "Docker": "Docker Certified Associate"}


def fmt_years(y: float | None) -> str:
    if y is None:
        return "-"
    return f"{int(y)}" if float(y).is_integer() else f"{y:.1f}"


# ---------------------------------------------------------------------------
# Requirement extraction (used by the extract_requirements tool)
# ---------------------------------------------------------------------------
_HDR_MUST = re.compile(r"^(must[- ]?haves?|requirements?|required(\s+\w+)?|minimum qualifications|basic qualifications|"
                       r"qualifications|what you('ll| will)? need|you (have|bring|must have)|key skills|skills( required)?|"
                       r"technical requirements|essential)\b", re.I)
_HDR_NICE = re.compile(r"^(nice[- ]to[- ]haves?|preferred(\s+\w+)?|bonus( points)?|good[- ]to[- ]haves?|pluses|desired|"
                       r"additional (skills|qualifications)|extra credit)\b", re.I)
_HDR_OTHER = re.compile(r"^(about|responsibilities|what you('ll| will)? do|the role|benefits|perks|who we are|company|"
                        r"role overview|overview|our team|education)\b", re.I)
NICE_MARK = re.compile(r"\b(nice[- ]to[- ]have|preferred|preferably|a plus|bonus|good[- ]to[- ]have|ideally|desirable|"
                       r"optional|familiarity with|exposure to)\b", re.I)
MUST_MARK = re.compile(r"\b(must|required|mandatory|essential|need to have|non[- ]negotiable)\b", re.I)
YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)", re.I)
_BULLET = re.compile(r"^\s*([-*•▪◦]|\d+[.)])\s*")


def user_skills(text: str) -> list[str]:
    """Skill extraction for short user-typed text: permissive, but 'go'/'rest' must look like the tech."""
    skills = extract_skills(text, strict=False)
    if "Go" in skills and not re.search(r"\b(Go|GO|golang)\b", text):
        skills.remove("Go")
    if "REST APIs" in skills and not re.search(r"\bREST\b|\brest ?api|\brestful", text, re.I) or (
            "REST APIs" in skills and re.search(r"\bthe rest\b", text, re.I)):
        skills.remove("REST APIs")
    return skills


def split_clauses(line: str) -> list[str]:
    parts = re.split(r"[;,]|\s+(?:but|while|although)\s+|\.\s+", line)
    return [p.strip() for p in parts if p and p.strip()]


def _assign_years(clause: str, skills: list[str]) -> tuple[dict[str, float], float | None]:
    """Decide whether 'N years' in a clause refers to a specific skill or to overall experience."""
    per_skill: dict[str, float] = {}
    overall = None
    for m in YEARS_RE.finditer(clause):
        n = float(m.group(1))
        after = clause[m.end(): m.end() + 30]
        after_skills = [s for s in extract_skills(after, strict=False) if s in skills]
        before = clause[max(0, m.start() - 40): m.start()]
        if after_skills and re.match(r"\s*(of|in|with|using|working with|hands-on)?\s*", after):
            per_skill[after_skills[0]] = n
        elif before.rstrip().endswith("(") and [s for s in extract_skills(before, strict=False) if s in skills]:
            per_skill[[s for s in extract_skills(before, strict=False) if s in skills][-1]] = n
        else:
            overall = max(overall or 0, n)
    return per_skill, overall


def parse_requirements(text: str, known_locations: list[str] | None = None) -> dict[str, Any]:
    """Split a JD (or a natural-language query) into must-have vs nice-to-have requirements."""
    lines = [l for l in text.splitlines()]
    title = next((l.strip() for l in lines if l.strip()), "Untitled role")
    title = re.split(r"\s+[—–|]\s+|\s+-\s+(?=[A-Z])", title)[0].strip()[:80]

    has_headers = any(_HDR_MUST.match(_BULLET.sub("", l).strip().rstrip(":")) or
                      _HDR_NICE.match(_BULLET.sub("", l).strip().rstrip(":")) for l in lines if len(l.split()) <= 5)
    mode = "none" if has_headers else "must"

    must: dict[str, float | None] = {}
    nice: list[str] = []
    min_years: float | None = None
    education = None

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        bare = _BULLET.sub("", line).strip()
        is_bullet = bool(_BULLET.match(line))
        hdr = bare.rstrip(":").strip()
        if not is_bullet and len(hdr.split()) <= 5 and i > 0:
            if _HDR_NICE.match(hdr):
                mode = "nice"; continue
            if _HDR_MUST.match(hdr):
                mode = "must"; continue
            if _HDR_OTHER.match(hdr):
                mode = "education" if hdr.lower().startswith("education") else "none"; continue
        if i == 0 and has_headers:
            continue  # title line

        deg = re.search(r"\b(ph\.?d|doctorate|master'?s?|m\.?s\.?|m\.?tech|bachelor'?s?|b\.?s\.?|b\.?tech|b\.?e\.?|degree)\b", bare, re.I)
        if deg and (mode == "education" or re.search(r"degree|bachelor|master|ph\.?d", bare, re.I)):
            word = deg.group(1).lower()
            level = "PhD" if word.startswith(("ph", "doc")) else "Masters" if word.startswith(("m", )) else "Bachelors"
            education = {"level": level, "required": not NICE_MARK.search(bare)}
            if mode == "education":
                continue

        clauses = split_clauses(bare)
        first_marker = ("nice" if NICE_MARK.search(clauses[0]) else "must" if MUST_MARK.search(clauses[0]) else None) if clauses else None
        line_mode = first_marker or (mode if mode in ("must", "nice") else "nice")  # skills under "About/Responsibilities" are implicit pluses
        for clause in clauses:
            c_mode = "nice" if NICE_MARK.search(clause) else "must" if MUST_MARK.search(clause) else line_mode
            skills = extract_skills(clause, strict=len(clause.split()) > 4) if len(clause.split()) > 4 else user_skills(clause)
            per_skill, overall = _assign_years(clause, skills)
            if overall is not None and (not skills or not per_skill):
                if not skills or re.search(r"experience|professional|industry|overall|total|engineering", clause, re.I):
                    min_years = max(min_years or 0, overall)
            for s in skills:
                if c_mode == "must":
                    must[s] = per_skill.get(s, must.get(s))
                elif s not in must and s not in nice:
                    nice.append(s)

    nice = [s for s in nice if s not in must]
    location = None
    if known_locations:
        for loc in known_locations:
            if re.search(rf"\b(in|from|based in|located in|near)\s+{re.escape(loc)}\b", text, re.I) or (
                    loc.lower() == "remote" and re.search(r"\bremote\b", text, re.I)):
                location = [loc]
                break
    return {
        "title": title,
        "must_have": [{"skill": s, "min_years": y} for s, y in must.items()],
        "nice_to_have": [{"skill": s} for s in nice],
        "min_years": min_years,
        "education": education,
        "location": location,
        "weights": dict(DEFAULT_WEIGHTS),
    }


def requirements_summary(reqs: dict, multiline: bool = False) -> str:
    must = ", ".join(f"{m['skill']}" + (f" ({fmt_years(m['min_years'])}+ yrs)" if m.get("min_years") else "")
                     for m in reqs["must_have"]) or "—"
    nice = ", ".join(n["skill"] for n in reqs["nice_to_have"]) or "—"
    parts = [f"**Must-have:** {must}", f"**Nice-to-have:** {nice}",
             f"**Experience:** {fmt_years(reqs['min_years'])}+ yrs" if reqs.get("min_years") else "**Experience:** any"]
    if reqs.get("education"):
        e = reqs["education"]
        parts.append(f"**Education:** {e['level']} ({'required' if e['required'] else 'preferred'})")
    if reqs.get("location"):
        parts.append(f"**Location:** {', '.join(reqs['location'])}")
    w = reqs.get("weights", DEFAULT_WEIGHTS)
    if w != DEFAULT_WEIGHTS:
        tot = sum(w.values()) or 1
        parts.append("**Weights:** " + ", ".join(f"{COMPONENT_LABELS[k].split(' (')[0]} {v / tot:.0%}" for k, v in w.items() if v))
    return ("\n".join(f"- {p}" for p in parts)) if multiline else " · ".join(parts)


# ---------------------------------------------------------------------------
# Round 1: fast scoring of every resume
# ---------------------------------------------------------------------------
def score_candidate(r: Resume, reqs: dict, relevance: float, assume_skills: frozenset = frozenset()) -> dict:
    have = r.all_skills() | set(assume_skills)
    weights = {**DEFAULT_WEIGHTS, **(reqs.get("weights") or {})}
    comps: dict[str, float] = {}
    matched, missing, short = [], [], []

    if reqs["must_have"]:
        vals = []
        for m in reqs["must_have"]:
            s, need = m["skill"], m.get("min_years")
            if s in have:
                yrs = r.skill_years(s) if s not in assume_skills else (need or 1.0)
                # no explicit minimum: reward demonstrated use over a bare skills-list mention
                v = 1.0 if need else 0.8 + 0.2 * min(1.0, yrs / 3)
                if need and yrs < need:
                    v = max(0.4, yrs / need)
                    short.append({"skill": s, "have": yrs, "need": need})
                matched.append({"skill": s, "years": yrs, "recent": r.skill_recent(s)})
            else:
                v = 0.0
                missing.append(s)
            vals.append(v)
        comps["must_have"] = sum(vals) / len(vals)

    min_y = reqs.get("min_years") or 0
    comps["experience"] = min(1.0, r.total_years / min_y) if min_y else min(1.0, 0.4 + r.total_years / 10)

    nice_names = [n["skill"] for n in reqs["nice_to_have"]]
    matched_nice = [s for s in nice_names if s in have]
    if nice_names:
        comps["nice_to_have"] = len(matched_nice) / len(nice_names)

    comps["relevance"] = float(relevance)

    degree = r.highest_degree()
    if reqs.get("education"):
        need = DEGREE_LEVEL[reqs["education"]["level"]]
        lvl = DEGREE_LEVEL.get(degree, 0)
        comps["education"] = 1.0 if lvl >= need else 0.5 if lvl == need - 1 else 0.2

    active = {k: weights.get(k, 0) for k in comps if weights.get(k, 0) > 0}
    wsum = sum(active.values()) or 1.0
    contributions = {k: round(comps[k] * w / wsum * 100, 2) for k, w in active.items()}
    pre = sum(contributions.values())
    total = pre * (MISSING_MUST_PENALTY ** len(missing))

    excluded = bool(reqs.get("location")) and not any(
        loc.lower() in (r.location or "").lower() for loc in reqs["location"])

    return {
        "candidate_id": r.candidate_id, "name": r.name, "title": r.title, "location": r.location,
        "score": round(total, 1), "components": {k: round(v, 3) for k, v in comps.items()},
        "contributions": contributions, "penalty": round(pre - total, 2),
        "matched_must": matched, "missing_must": missing, "short_years": short,
        "matched_nice": matched_nice, "missing_nice": [s for s in nice_names if s not in have],
        "total_years": r.total_years, "degree": degree, "relevance": round(float(relevance), 3),
        "excluded": excluded,
    }


def one_line_reason(b: dict, reqs: dict) -> str:
    n_must = len(reqs["must_have"])
    bits = []
    if n_must:
        bits.append(f"{len(b['matched_must'])}/{n_must} must-haves")
    if b["missing_must"]:
        bits.append("missing " + ", ".join(b["missing_must"]))
    for s in b["short_years"]:
        bits.append(f"{s['skill']} {fmt_years(s['have'])}/{fmt_years(s['need'])} yrs")
    bits.append(f"{fmt_years(b['total_years'])} yrs exp")
    if reqs["nice_to_have"]:
        bits.append(f"{len(b['matched_nice'])}/{len(reqs['nice_to_have'])} nice-to-haves")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# Round 2: deep evidence analysis
# ---------------------------------------------------------------------------
_QUANT = re.compile(r"\d+\s*%|\$\s?\d|\d+\s*[MK]\b|\d+M\b|\b\d{2,}\s+(services|users|engineers|customers)", re.I)
_LEAD = re.compile(r"\b(led|lead|mentor(ed)?|architected|managed|ran code reviews|owned|set engineering standards|"
                   r"partnered with stakeholders)\b", re.I)


def deep_profile(r: Resume, reqs: dict) -> dict:
    bullets = [b for role in r.roles for b in role.bullets]
    quantified = [b for b in bullets if _QUANT.search(b)]
    leadership = [b for b in bullets if _LEAD.search(b)]
    role_skill_set = {s for role in r.roles for s in role.skills}

    evidence = []
    for m in reqs["must_have"]:
        s = m["skill"]
        evidence.append({
            "skill": s,
            "listed": s in r.skills,
            "years": r.skill_years(s),
            "recent": r.skill_recent(s),
            "demonstrated": s in role_skill_set,
            "need": m.get("min_years"),
        })

    gaps = []
    chrono = sorted(r.roles, key=lambda x: x.start_year)
    for a, b in zip(chrono, chrono[1:]):
        if b.start_year - a.end_year >= 1:
            gaps.append((a.end_year, b.start_year))
    avg_tenure = (r.total_years / len(r.roles)) if r.roles else 0.0
    req_skills = {m["skill"] for m in reqs["must_have"]} | {n["skill"] for n in reqs["nice_to_have"]}
    rel_certs = [c for c in r.certifications if any(s.lower() in c.lower() for s in req_skills) or
                 any(alias in c.lower() for s in req_skills for alias in SKILL_TAXONOMY.get(s, ("", []))[1])]

    return {
        "evidence": evidence,
        "n_bullets": len(bullets),
        "quantified": quantified,
        "impact_ratio": round(len(quantified) / len(bullets), 2) if bullets else 0.0,
        "leadership": leadership,
        "avg_tenure": round(avg_tenure, 1),
        "job_hopping": len(r.roles) >= 3 and avg_tenure < 1.5,
        "career_gaps": gaps,
        "relevant_certs": rel_certs,
        "claimed_not_demonstrated": [e["skill"] for e in evidence if e["listed"] and not e["demonstrated"]],
        "current_role": next((f"{x.title} @ {x.company}" for x in r.roles if x.current), r.title),
    }


def deep_score(initial: dict, prof: dict, reqs: dict) -> dict:
    """Blend the round-1 score with evidence quality from round 2 (0-100)."""
    ev = prof["evidence"]
    if ev:
        depth = []
        for e in ev:
            if not (e["listed"] or e["demonstrated"]):
                depth.append(0.0); continue
            target = e["need"] or 2.0
            d = min(1.0, e["years"] / target) if e["demonstrated"] else 0.35
            depth.append(min(1.0, d + (0.1 if e["recent"] else 0)))
        evidence_depth = sum(depth) / len(depth)
    else:
        evidence_depth = 0.6
    impact = min(1.0, prof["impact_ratio"] / 0.5)
    senior = (reqs.get("min_years") or 0) >= 5
    leadership = min(1.0, len(prof["leadership"]) / 2) if senior else 0.7 + 0.3 * min(1.0, len(prof["leadership"]))
    stability = 0.4 if prof["job_hopping"] else (0.8 if prof["career_gaps"] else 1.0)
    extras = min(1.0, 0.5 * len(prof["relevant_certs"]) + (len(initial["matched_nice"]) / max(1, len(reqs["nice_to_have"]))) * 0.7)
    min_y = reqs.get("min_years") or 0
    seniority = min(1.0, initial["total_years"] / min_y) if min_y else 1.0
    parts = {"evidence_depth": evidence_depth, "seniority": seniority, "impact": impact, "leadership": leadership,
             "stability": stability, "certs_and_extras": extras}
    w = {"evidence_depth": 0.4, "seniority": 0.15, "impact": 0.15, "leadership": 0.1, "stability": 0.1, "certs_and_extras": 0.1}
    evidence_score = sum(parts[k] * w[k] for k in w) * 100
    final = 0.5 * initial["score"] + 0.5 * evidence_score * (MISSING_MUST_PENALTY ** len(initial["missing_must"]))
    return {"evidence_parts": {k: round(v, 2) for k, v in parts.items()}, "evidence_score": round(evidence_score, 1),
            "final_score": round(final, 1)}


# ---------------------------------------------------------------------------
# Round 3: decision
# ---------------------------------------------------------------------------
def hire_decision(initial: dict, deep: dict, reqs: dict | None = None) -> dict:
    s = deep["final_score"]
    missing = initial["missing_must"]
    min_y = (reqs or {}).get("min_years") or 0
    far_below_bar = bool(min_y) and initial["total_years"] < 0.6 * min_y
    if far_below_bar:
        verdict = "Borderline (Lean No Hire)" if s >= HIRE and not missing else "No Hire"
    elif s >= STRONG_HIRE and not missing:
        verdict = "Strong Hire"
    elif s >= HIRE and not missing:
        verdict = "Hire"
    elif s >= BORDERLINE or (s >= HIRE and len(missing) == 1):
        verdict = "Borderline (Lean No Hire)"
    else:
        verdict = "No Hire"
    nearest = min(abs(s - t) for t in (STRONG_HIRE, HIRE, BORDERLINE))
    confidence = "High" if nearest >= 6 else "Medium" if nearest >= 3 else "Low"
    return {"verdict": verdict, "confidence": confidence, "final_score": s}


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def strengths_and_gaps(r: Resume, b: dict, reqs: dict, prof: dict | None = None) -> tuple[list[str], list[str]]:
    prof = prof or deep_profile(r, reqs)
    strengths, gaps = [], []
    for m in sorted(b["matched_must"], key=lambda x: -x["years"]):
        need = next((x.get("min_years") for x in reqs["must_have"] if x["skill"] == m["skill"]), None)
        if m["years"] and (not need or m["years"] >= need):
            strengths.append(f"{m['skill']}: {fmt_years(m['years'])} yrs hands-on" + (" (current role)" if m["recent"] else "")
                             + (f", meets {fmt_years(need)}+ requirement" if need else ""))
    min_y = reqs.get("min_years")
    if min_y and b["total_years"] >= min_y:
        strengths.append(f"{fmt_years(b['total_years'])} yrs total experience (requirement {fmt_years(min_y)}+)")
    if b["matched_nice"]:
        strengths.append("Nice-to-haves: " + ", ".join(b["matched_nice"]))
    if len(prof["quantified"]) >= 2:
        strengths.append(f"Quantifies impact ({len(prof['quantified'])} measurable achievements)")
    if prof["leadership"]:
        strengths.append("Leadership signals: " + prof["leadership"][0][:90])
    if prof["relevant_certs"]:
        strengths.append("Certified: " + "; ".join(prof["relevant_certs"]))

    for s in b["missing_must"]:
        gaps.append(f"Missing must-have: {s}")
    for s in b["short_years"]:
        gaps.append(f"{s['skill']}: {fmt_years(s['have'])} yrs vs {fmt_years(s['need'])}+ required")
    if min_y and b["total_years"] < min_y:
        gaps.append(f"Experience {fmt_years(b['total_years'])} yrs vs {fmt_years(min_y)}+ required")
    for s in prof["claimed_not_demonstrated"]:
        gaps.append(f"{s} listed in skills but not evidenced in any role")
    if b["missing_nice"]:
        gaps.append("No " + ", ".join(b["missing_nice"][:4]) + (" (nice-to-have)" if len(b["missing_nice"]) == 1 else " (nice-to-haves)"))
    if reqs.get("education") and b.get("components", {}).get("education", 1) < 1:
        gaps.append(f"Education: {b['degree']} vs {reqs['education']['level']} {'required' if reqs['education']['required'] else 'preferred'}")
    if prof["n_bullets"] and not prof["quantified"]:
        gaps.append("No quantified outcomes in work history")
    if prof["job_hopping"]:
        gaps.append(f"Short average tenure ({fmt_years(prof['avg_tenure'])} yrs/role)")
    for a, c in prof["career_gaps"]:
        gaps.append(f"Career gap {a}–{c}")
    return strengths, gaps


def improvement_suggestions(r: Resume, b: dict, reqs: dict, prof: dict | None = None) -> list[str]:
    """Concrete ways a borderline candidate could close the gap (also useful to recruiters as interview focus)."""
    prof = prof or deep_profile(r, reqs)
    have = r.all_skills()
    out = []
    for s in b["missing_must"]:
        related = [x for x in have if skill_category(x) == skill_category(s) and x != s]
        tip = f"Gain hands-on {s} experience (a production feature or substantial side project)"
        if related:
            tip += f"; adjacent experience in {', '.join(sorted(related)[:3])} should transfer quickly"
        if s in CERT_FOR_SKILL:
            tip += f"; the {CERT_FOR_SKILL[s]} would be credible proof"
        out.append(tip + ".")
    for s in b["short_years"]:
        out.append(f"Needs ~{fmt_years(round(s['need'] - s['have'], 1))} more year(s) of {s['skill']}; "
                   f"consider for a mid-level opening now, or probe depth in the interview.")
    min_y = reqs.get("min_years")
    if min_y and b["total_years"] < min_y:
        out.append(f"Below the {fmt_years(min_y)}+ yrs bar — a fit for a less senior level of this role.")
    for s in prof["claimed_not_demonstrated"]:
        out.append(f"Add concrete project evidence for {s} (what was built, scale, outcome).")
    if prof["n_bullets"] and prof["impact_ratio"] < 0.3:
        out.append("Quantify achievements (latency, revenue, users, cost) — currently few measurable outcomes.")
    for s in b["missing_nice"][:2]:
        out.append(f"Picking up {s} would strengthen the profile (nice-to-have).")
    if (reqs.get("min_years") or 0) >= 5 and not prof["leadership"]:
        out.append("Show ownership/leadership (leading a project, mentoring) expected at senior level.")
    return out or ["No material gaps — focus the interview on depth and culture fit."]


def explain_pair(a: dict, b: dict, ra: Resume, rb: Resume, reqs: dict, rank_a: int | None, rank_b: int | None,
                 what_if_score: float | None = None) -> str:
    """Why does candidate a rank above candidate b? (a is expected to be the higher one.)"""
    hi, lo = (a, b) if a["score"] >= b["score"] else (b, a)
    rhi, rlo = (ra, rb) if hi is a else (rb, ra)
    rank_hi, rank_lo = (rank_a, rank_b) if hi is a else (rank_b, rank_a)
    lines = [f"**{hi['name']}** ({hi['score']}, rank #{rank_hi}) ranks above **{lo['name']}** "
             f"({lo['score']}, rank #{rank_lo}) — a {hi['score'] - lo['score']:.1f}-point gap.\n",
             "| Component | " + hi["name"] + " | " + lo["name"] + " | Δ points |", "|---|---|---|---|"]
    deltas = []
    for k in hi["contributions"]:
        dh, dl = hi["contributions"].get(k, 0), lo["contributions"].get(k, 0)
        deltas.append((dh - dl, k))
        lines.append(f"| {COMPONENT_LABELS[k]} | {dh:.1f} | {dl:.1f} | {dh - dl:+.1f} |")
    if hi["penalty"] or lo["penalty"]:
        lines.append(f"| Missing must-have penalty | −{hi['penalty']:.1f} | −{lo['penalty']:.1f} | {lo['penalty'] - hi['penalty']:+.1f} |")
        deltas.append((lo["penalty"] - hi["penalty"], "penalty"))
    lines.append("")

    deltas.sort(reverse=True)
    lines.append("**Decisive factors:**")
    n_must, n_nice = len(reqs["must_have"]), len(reqs["nice_to_have"])
    for d, k in deltas:
        if d <= 0.3:
            continue
        if k in ("must_have", "penalty"):
            if k == "penalty" and any(x[1] == "must_have" and x[0] > 0.3 for x in deltas):
                continue  # already explained by the must-have line
            txt = f"Must-haves: {hi['name']} covers {len(hi['matched_must'])}/{n_must}"
            if lo["missing_must"]:
                txt += f"; {lo['name']} is missing {', '.join(lo['missing_must'])}"
            if lo["short_years"]:
                txt += "; " + ", ".join(f"{lo['name']} has {fmt_years(s['have'])} yrs {s['skill']} vs {fmt_years(s['need'])}+ needed" for s in lo["short_years"])
            lines.append(f"- {txt} (+{d:.1f})")
        elif k == "experience":
            lines.append(f"- Experience: {fmt_years(hi['total_years'])} vs {fmt_years(lo['total_years'])} yrs"
                         + (f" against a {fmt_years(reqs['min_years'])}+ bar" if reqs.get("min_years") else "") + f" (+{d:.1f})")
        elif k == "nice_to_have":
            lines.append(f"- Nice-to-haves: {len(hi['matched_nice'])}/{n_nice} ({', '.join(hi['matched_nice']) or '—'}) vs "
                         f"{len(lo['matched_nice'])}/{n_nice} ({', '.join(lo['matched_nice']) or '—'}) (+{d:.1f})")
        elif k == "relevance":
            lines.append(f"- Resume text is a closer semantic match to the role ({hi['relevance']:.2f} vs {lo['relevance']:.2f}) (+{d:.1f})")
        elif k == "education":
            lines.append(f"- Education: {hi['degree']} vs {lo['degree']} (+{d:.1f})")
    counter = [(d, k) for d, k in deltas if d < -0.3]
    if counter:
        lines.append(f"\n**Where {lo['name']} is stronger:**")
        for d, k in counter:
            if k == "nice_to_have":
                lines.append(f"- Nice-to-haves: {', '.join(lo['matched_nice'])} ({-d:.1f} pts)")
            elif k == "experience":
                lines.append(f"- More experience: {fmt_years(lo['total_years'])} yrs ({-d:.1f} pts)")
            elif k == "relevance":
                lines.append(f"- Closer resume wording match ({-d:.1f} pts)")
            elif k == "education":
                lines.append(f"- Education: {lo['degree']} ({-d:.1f} pts)")
            elif k == "must_have":
                lines.append(f"- Must-have depth ({-d:.1f} pts)")
    if what_if_score is not None and lo["missing_must"]:
        verdict = "would overtake" if what_if_score > hi["score"] else "would still trail"
        lines.append(f"\n**What would flip it:** with {', '.join(lo['missing_must'])}, {lo['name']} would score ≈{what_if_score} "
                     f"and {verdict} {hi['name']}.")
    return "\n".join(lines)


def _req_diff(old: dict, new: dict) -> dict:
    om = {m["skill"]: m.get("min_years") for m in old["must_have"]}
    nm = {m["skill"]: m.get("min_years") for m in new["must_have"]}
    on = {n["skill"] for n in old["nice_to_have"]}
    nn = {n["skill"] for n in new["nice_to_have"]}
    return {
        "added_must": [s for s in nm if s not in om],
        "removed_must": [s for s in om if s not in nm],
        "must_years_changed": [s for s in nm if s in om and nm[s] != om[s]],
        "added_nice": [s for s in nn if s not in on],
        "removed_nice": [s for s in on if s not in nn],
        "min_years": (old.get("min_years"), new.get("min_years")) if old.get("min_years") != new.get("min_years") else None,
        "weights": [k for k in new.get("weights", {}) if new["weights"].get(k) != old.get("weights", {}).get(k)],
        "education": old.get("education") != new.get("education"),
        "location": old.get("location") != new.get("location"),
    }


def explain_rank_changes(old_scores: dict, new_scores: dict, old_order: list[str], new_order: list[str],
                         old_reqs: dict, new_reqs: dict, top_k: int) -> list[dict]:
    diff = _req_diff(old_reqs, new_reqs)
    old_rank = {c: i + 1 for i, c in enumerate(old_order)}
    new_rank = {c: i + 1 for i, c in enumerate(new_order)}
    watch = list(dict.fromkeys(new_order[:top_k] + [c for c in old_order[:top_k] if c in new_rank]))
    changes = []
    for cid in watch:
        o, n = old_rank.get(cid), new_rank.get(cid)
        if o == n:
            continue
        ob, nb = old_scores.get(cid), new_scores[cid]
        have = {m["skill"]: m["years"] for m in nb["matched_must"]}
        yrs = lambda s: f"{fmt_years(have[s])} yrs" if have.get(s) else "listed, no role evidence"
        signed: list[tuple[int, str]] = []  # (+1 helps / -1 hurts, reason)
        for s in diff["added_must"]:
            signed.append((1, f"has newly required {s} ({yrs(s)})") if s in have else (-1, f"lacks newly required {s}"))
        for s in diff["removed_must"]:
            if ob and s in ob["missing_must"]:
                signed.append((1, f"no longer penalised for missing {s}"))
            elif ob and s in [m["skill"] for m in ob["matched_must"]]:
                signed.append((-1, f"their {s} no longer counts as a must-have"))
        new_need = {m["skill"]: m.get("min_years") for m in new_reqs["must_have"]}
        for s in diff["must_years_changed"]:
            if new_need.get(s) is None:
                signed.append((1 if s in have and any(x["skill"] == s for x in (ob or {}).get("short_years", [])) else 0,
                               f"{s} years minimum removed"))
            elif any(x["skill"] == s for x in nb["short_years"]):
                signed.append((-1, f"below the new {fmt_years(new_need[s])}+ yr {s} bar ({yrs(s)})"))
            elif s in have:
                signed.append((1, f"meets the new {fmt_years(new_need[s])}+ yr {s} bar ({yrs(s)})"))
        for s in diff["added_nice"]:
            if s in nb["matched_nice"]:
                signed.append((1, f"has new nice-to-have {s}"))
        if diff["min_years"]:
            _, new_min = diff["min_years"]
            ok = nb["total_years"] >= (new_min or 0)
            signed.append((1 if ok else -1, f"{fmt_years(nb['total_years'])} yrs experience vs new {fmt_years(new_min or 0)}+ bar"))
        for k in diff["weights"]:
            v = nb["components"].get(k, 0)
            signed.append((1 if v >= 0.8 else -1 if v <= 0.5 else 0, f"{COMPONENT_LABELS.get(k, k)} reweighted (scores {v:.2f} on it)"))
        if diff["location"] and nb["excluded"]:
            signed.append((-1, f"filtered out by location ({nb['location']})"))
        # Prefer reasons that point the same way as the move; keep it to the three most relevant.
        moved_up = (o or 10_000) > (n or 10_000)
        want = 1 if moved_up else -1
        signed.sort(key=lambda x: 0 if x[0] == want else 1 if x[0] == 0 else 2)
        reasons = [r for _, r in signed[:3]]
        if not any(sg == want for sg, _ in signed) and signed:
            reasons.append("others gained more from the change" if not moved_up else "others lost more from the change")
        if not reasons and ob:
            deltas = sorted(((nb["contributions"].get(k, 0) - ob["contributions"].get(k, 0), k) for k in nb["contributions"]),
                            key=lambda x: -abs(x[0]))
            if deltas and abs(deltas[0][0]) > 0.2:
                reasons.append(f"{COMPONENT_LABELS[deltas[0][1]]} {deltas[0][0]:+.1f} pts")
            else:
                reasons.append("others moved around them")
        changes.append({
            "candidate_id": cid, "name": nb["name"], "old_rank": o, "new_rank": n,
            "old_score": ob["score"] if ob else None, "new_score": nb["score"],
            "delta": (o - n) if o and n else None,
            "entered_top": n is not None and n <= top_k and (o is None or o > top_k),
            "left_top": o is not None and o <= top_k and (n is None or n > top_k),
            "reason": "; ".join(dict.fromkeys(reasons)),
        })
    changes.sort(key=lambda c: (c["new_rank"] or 999))
    return changes
