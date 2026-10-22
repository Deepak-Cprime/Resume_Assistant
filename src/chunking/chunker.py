"""Turn one CandidateProfile into chunks, following the Phase 1 design:

    level 0  profile      headline + seniority/YOE + summary + top skills (embedded)
    level 1  sections     skills, education, certifications, projects ... (embedded); short
                          neighbours are merged; experience is stored whole but NOT embedded
    level 2  entries      one chunk per job (embedded), child of the experience section;
                          an entry over its budget splits on bullets and repeats its header

Every embedded chunk carries a structural prefix (candidate ID, section, title @ company, dates)
and skill tags. IDs are deterministic: the same profile and budgets always give the same chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.chunking import enrich
from src.chunking.budgets import MIN_BODY_TOKENS, Budgets
from src.chunking.tokens import estimate_tokens
from src.metadata.schema import CandidateProfile, Experience
from src.metadata.skills import SkillVocabulary, default_vocabulary
from src.structure.text import is_bullet, strip_md

Counter = Callable[[str], int]
SKIP_SECTIONS = {"header", "contact", "summary", "experience", "references"}
TOP_SKILLS = 12
NEWLINE_SLACK = 4  # tokens kept free for line breaks between prefix, body and tags
NOT_EMBEDDED_LIMIT = 10**9  # stored parents are not embedded, so no token limit applies


def chunk_id_for(key: str) -> int:
    """60-bit integer from the readable key (FAISS needs integer ids; SQLite needs < 2**63)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)


@dataclass
class Chunk:
    key: str
    candidate_id: int
    doc_id: str
    level: int
    section: str
    text: str
    embed_text: str
    embed: bool
    n_tokens: int
    parent_key: str | None = None
    entry_title: str = ""
    start: str | None = None
    end: str | None = None
    skills: list[str] = field(default_factory=list)
    page: int | None = None

    @property
    def chunk_id(self) -> int:
        return chunk_id_for(self.key)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()

    def to_row(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_key": self.key,
            "candidate_id": self.candidate_id,
            "doc_id": self.doc_id,
            "level": self.level,
            "parent_id": chunk_id_for(self.parent_key) if self.parent_key else None,
            "section": self.section,
            "entry_title": self.entry_title,
            "start": self.start,
            "end": self.end,
            "skills": json.dumps(self.skills),
            "text": self.text,
            "embed_text": self.embed_text,
            "embed": int(self.embed),
            "n_tokens": self.n_tokens,
            "page": self.page,
            "content_hash": self.content_hash,
        }


# --- splitting and packing --------------------------------------------------------------


def split_oversize(unit: str, budget: int, count: Counter) -> list[str]:
    """A unit larger than the budget is cut at sentence ends, then at words."""
    if count(unit) <= budget:
        return [unit]
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", unit):
        if count(sentence) <= budget:
            out.append(sentence)
            continue
        current: list[str] = []
        for word in sentence.split():
            if current and count(" ".join([*current, word])) > budget:
                out.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            out.append(" ".join(current))
    return out


def pack(units: list[str], budget: int, min_tok: int, count: Counter) -> list[str]:
    """Greedily fill parts up to `budget` without cutting a unit (a bullet).

    Greedy filling can leave a tiny last part. When it is smaller than `min_tok`, bullets are
    moved to it from the end of the previous part, as long as neither part breaks the budget
    or the previous part drops below `min_tok`."""
    parts: list[list[str]] = []
    current: list[str] = []
    used = 0
    for piece in (p for u in units for p in split_oversize(u, budget, count)):
        cost = count(piece)
        if current and used + cost > budget:
            parts.append(current)
            current, used = [], 0
        current.append(piece)
        used += cost
    if current:
        parts.append(current)

    if len(parts) > 1:
        previous, last = parts[-2], parts[-1]
        size = lambda p: sum(count(u) for u in p)  # noqa: E731 - tiny local helper
        while len(previous) > 1 and size(last) < min_tok:
            moved = count(previous[-1])
            if size(last) + moved > budget or size(previous) - moved < min_tok:
                break
            last.insert(0, previous.pop())
    return ["\n".join(p) for p in parts]


# --- profile -> units -------------------------------------------------------------------


def entry_units(exp: Experience) -> list[str]:
    return ([exp.description] if exp.description else []) + [f"- {b}" for b in exp.bullets]


def plain_section(text: str) -> str:
    """Section Markdown as plain lines, keeping '- ' bullets."""
    lines = []
    for raw in text.splitlines():
        line = strip_md(raw)
        if line:
            lines.append(f"- {line}" if is_bullet(raw) else line)
    return "\n".join(lines)


def section_texts(profile: CandidateProfile) -> list[tuple[str, str]]:
    """(label, plain text) per chunkable non-experience section, in document order, with
    repeated labels combined and unrecognised sections labelled 'other'."""
    merged: dict[str, list[str]] = {}
    for s in profile.sections:
        if s.label in SKIP_SECTIONS:
            continue
        label = "other" if s.label == "unknown" else s.label
        plain = plain_section(s.text)
        if plain:
            merged.setdefault(label, []).append(plain)
    return [(label, "\n".join(texts)) for label, texts in merged.items()]


def unit_lengths(
    profile: CandidateProfile, count: Counter = estimate_tokens
) -> dict[str, list[int]]:
    """Token length of every chunkable unit, by section type; the input to budget statistics."""
    out: dict[str, list[int]] = {}
    for exp in profile.experiences:
        n = count("\n".join(entry_units(exp)))
        if n:
            out.setdefault("experience", []).append(n)
    for label, plain in section_texts(profile):
        out.setdefault(label, []).append(count(plain))
    return out


def profile_text(profile: CandidateProfile) -> str:
    summary = " ".join(
        plain_section(s.text).replace("\n", " ") for s in profile.sections if s.label == "summary"
    )
    ranked = sorted(profile.skills, key=lambda s: (not s.used, -(s.years or 0), not s.claimed))
    lines = [
        profile.headline,
        f"{profile.seniority.title()} level, {profile.total_yoe:.1f} years of experience"
        if profile.total_yoe else "",
        summary,
        f"Top skills: {', '.join(s.skill for s in ranked[:TOP_SKILLS])}" if ranked else "",
    ]
    return "\n".join(line for line in lines if line)


# --- builder ----------------------------------------------------------------------------


class _Builder:
    def __init__(
        self,
        profile: CandidateProfile,
        candidate_id: int,
        doc_id: str,
        budgets: Budgets,
        model_limit: int,
        count: Counter,
        vocab: SkillVocabulary,
    ) -> None:
        self.profile, self.cid, self.doc_id = profile, candidate_id, doc_id
        self.budgets, self.limit, self.count, self.vocab = budgets, model_limit, count, vocab
        self.names = [profile.name]
        self.chunks: list[Chunk] = []

    def room(self, max_tok: int, prefix: str, tags: str) -> int:
        """Body tokens available once the prefix and tags are paid for."""
        overhead = self.count(prefix) + self.count(tags) + NEWLINE_SLACK
        return max(MIN_BODY_TOKENS, min(max_tok, self.limit - overhead))

    def add(
        self, key: str, level: int, section: str, text: str, prefix: str, body: str,
        *, tags: str = "", embed: bool = True, **fields,
    ) -> None:
        limit = self.limit if embed else NOT_EMBEDDED_LIMIT
        embed_text = enrich.assemble_embed_text(
            prefix, body, tags, names=self.names, limit=limit, count=self.count
        )
        self.chunks.append(Chunk(
            key=key, candidate_id=self.cid, doc_id=self.doc_id, level=level, section=section,
            text=text, embed_text=embed_text, embed=embed,
            n_tokens=self.count(embed_text), **fields,
        ))

    def profile_chunk(self) -> None:
        text = profile_text(self.profile)
        if text:
            self.add(f"c{self.cid}-profile", 0, "profile", text,
                     enrich.section_prefix(self.cid, "profile"), text)

    def experience(self) -> None:
        if not self.profile.experiences:
            return
        parent_key = f"c{self.cid}-experience"
        budget = self.budgets.for_section("experience")
        parent_texts: list[str] = []
        entries: list[tuple[str, dict]] = []
        for i, exp in enumerate(self.profile.experiences, 1):
            units = entry_units(exp)
            header = enrich.entry_header(exp)
            parent_texts.append("\n".join([header, *units]).strip())
            skills = self.vocab.find("\n".join([exp.title, *units]))
            prefix = enrich.entry_prefix(self.cid, exp)
            tags = enrich.tags_line(skills)
            parts = pack(units, self.room(budget.max_tok, prefix, tags), budget.min_tok,
                         self.count) or [""]
            for p, body in enumerate(parts, 1):
                entries.append((f"c{self.cid}-exp-{i}-p{p}", {
                    "text": "\n".join(x for x in (header, body) if x),
                    "prefix": prefix, "body": body, "tags": tags,
                    "fields": {"parent_key": parent_key, "entry_title": enrich.entry_title(exp),
                               "start": exp.start,
                               "end": "present" if exp.is_current else exp.end,
                               "skills": skills},
                }))
        parent_text = "\n\n".join(parent_texts)
        self.add(parent_key, 1, "experience", parent_text,
                 enrich.section_prefix(self.cid, "experience"), parent_text, embed=False)
        for key, e in entries:
            self.add(key, 2, "experience", e["text"], e["prefix"], e["body"],
                     tags=e["tags"], **e["fields"])

    def sections(self) -> None:
        """Skills, education, certifications ... Sections below their min budget are merged
        with the next short section; a merged chunk never swallows a long section."""
        group: list[tuple[str, str]] = []
        for label, plain in section_texts(self.profile):
            if self.count(plain) < self.budgets.for_section(label).min_tok:
                group.append((label, plain))
                continue
            self._emit_group(group)
            group = []
            self._emit_group([(label, plain)])
        self._emit_group(group)

    def _emit_group(self, group: list[tuple[str, str]]) -> None:
        if not group:
            return
        label = "+".join(lbl for lbl, _ in group)
        text = (
            "\n\n".join(f"{enrich.section_title(lbl)}:\n{plain}" for lbl, plain in group)
            if len(group) > 1 else group[0][1]
        )
        prefix = enrich.section_prefix(self.cid, label)
        budget = self.budgets.for_section(label)
        parts = pack(text.splitlines(), self.room(budget.max_tok, prefix, ""), budget.min_tok,
                     self.count)
        base = f"c{self.cid}-{label.replace('+', '-')}"
        for p, body in enumerate(parts, 1):
            key = base if len(parts) == 1 else f"{base}-p{p}"
            self.add(key, 1, label, body, prefix, body)


def build_chunks(
    profile: CandidateProfile,
    *,
    candidate_id: int,
    doc_id: str,
    budgets: Budgets,
    model_limit: int,
    count: Counter = estimate_tokens,
    vocab: SkillVocabulary | None = None,
) -> list[Chunk]:
    builder = _Builder(profile, candidate_id, doc_id, budgets, model_limit, count,
                       vocab or default_vocabulary())
    builder.profile_chunk()
    builder.experience()
    builder.sections()
    return builder.chunks


def chunk_stats(chunks: list[Chunk], budgets: Budgets, model_limit: int) -> dict[str, float]:
    """Counts used by the Index page and the Phase 1 gates."""
    embedded = [c for c in chunks if c.embed]
    under = [c for c in embedded if c.n_tokens < budgets.for_section(c.section).min_tok]
    return {
        "chunks": len(chunks),
        "embedded": len(embedded),
        "over_limit": sum(c.n_tokens > model_limit for c in embedded),
        "under_min_share": len(under) / len(embedded) if embedded else 0.0,
        "avg_tokens": sum(c.n_tokens for c in embedded) / len(embedded) if embedded else 0.0,
    }
