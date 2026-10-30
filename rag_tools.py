"""
rag_tools.py — Milestone 2 RAG search over resumes.

A dependency-free retriever: resumes are chunked (profile / skills / each role),
indexed with BM25, and queries are expanded with canonical skill tokens so that
"ReactJS developer" retrieves resumes that say "React.js".

To use your own Milestone 2 retriever (e.g. a Chroma/FAISS vector store), replace
`ResumeIndex.search` or set `rag_tools.RETRIEVER = your_object` — anything with a
`.search(query: str, k: int) -> list[dict]` method returning
{"candidate_id", "name", "score" (0-1), "snippet"} works.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from langchain_core.tools import tool

from fs_tools import load_all_resumes
from skills import Resume, extract_skills

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.#+][a-z0-9]+)*[#+]*")
_STOP = set("""a an the and or of to in on for with at by from as is are was were be been this that
these those it its into over under using use used we our you your i my me per via""".split())


def _tokens(text: str, skill_list: bool = False) -> list[str]:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOP]
    skills = extract_skills(text, strict=not skill_list)
    # canonical skill tokens are boosted (x2) because they carry the most signal
    return words + [f"skill::{s.lower()}" for s in skills] * 2


class ResumeIndex:
    """BM25 index over resume chunks, aggregated to candidate level."""

    def __init__(self, resumes: dict[str, Resume], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.resumes = resumes
        self.chunks: list[tuple[str, str, list[str]]] = []  # (candidate_id, text, tokens)
        for cid, r in resumes.items():
            profile = f"{r.title}. {r.summary}"
            self.chunks.append((cid, profile, _tokens(profile)))
            if r.skills:
                txt = "Skills: " + ", ".join(r.skills)
                self.chunks.append((cid, txt, _tokens(txt, skill_list=True)))
            for role in r.roles:
                txt = f"{role.title} at {role.company}: " + " ".join(role.bullets)
                self.chunks.append((cid, txt, _tokens(txt)))
        self.N = len(self.chunks)
        self.avgdl = sum(len(t) for _, _, t in self.chunks) / max(1, self.N)
        df: Counter = Counter()
        for _, _, toks in self.chunks:
            df.update(set(toks))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(toks) for _, _, toks in self.chunks]

    def _bm25(self, q: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.chunks[i][2])
        s = 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 10) -> list[dict]:
        q = _tokens(query, skill_list=True)
        per_cand: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for i, (cid, text, _) in enumerate(self.chunks):
            sc = self._bm25(q, i)
            if sc > 0:
                per_cand[cid].append((sc, text))
        scored = []
        for cid, lst in per_cand.items():
            lst.sort(reverse=True)
            top = [s for s, _ in lst[:3]]
            agg = top[0] + 0.5 * sum(top[1:]) / max(1, len(top) - 1) if len(top) > 1 else top[0]
            scored.append((agg, cid, lst[0][1]))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 1.0
        return [
            {
                "candidate_id": cid,
                "name": self.resumes[cid].name,
                "score": round(s / best, 4),
                "snippet": snippet[:220] + ("…" if len(snippet) > 220 else ""),
            }
            for s, cid, snippet in scored[:k]
        ]


RETRIEVER: ResumeIndex | None = None


def get_retriever(rebuild: bool = False):
    global RETRIEVER
    if RETRIEVER is None or rebuild:
        RETRIEVER = ResumeIndex(load_all_resumes())
    return RETRIEVER


def semantic_search(query: str, k: int = 10) -> list[dict]:
    return get_retriever().search(query, k)


@tool
def rag_search(query: str, k: int = 10) -> list[dict]:
    """Semantic/keyword hybrid search over all resumes. Returns the k most relevant candidates
    with a 0-1 relevance score and the best-matching snippet."""
    return semantic_search(query, k)


RAG_TOOLS = [rag_search]
