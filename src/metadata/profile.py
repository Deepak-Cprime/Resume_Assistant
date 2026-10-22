"""Build a CandidateProfile from a resume's Markdown: sections, jobs, education, skills, YOE."""

from __future__ import annotations

from datetime import date

from src.metadata import llm_extract
from src.metadata.pii import email_hash
from src.metadata.regex_extract import (
    find_email,
    find_github,
    find_linkedin,
    find_location,
    find_name,
    find_phone,
)
from src.metadata.schema import (
    CandidateProfile,
    Contact,
    Experience,
    SectionInfo,
    SkillEvidence,
)
from src.metadata.skills import SkillVocabulary, default_vocabulary
from src.metadata.yoe import DEFAULT_GAP_MONTHS, compute_yoe, intervals, months_covered
from src.providers.base import LLMProvider
from src.structure.entries import parse_certifications, parse_education, parse_experiences
from src.structure.sections import (
    SectionMatcher,
    known_token_share,
    load_matcher,
    section_lines,
    split_sections,
)

MIN_KNOWN_TOKEN_SHARE = 0.9
SENIORITY_BY_TITLE = (
    ("director", ("director", "head of", "vp", "vice president", "chief")),
    ("principal", ("principal", "staff", "architect", "distinguished")),
    ("lead", ("lead", "manager")),
    ("senior", ("senior", "sr.", "sr ")),
)
MID_YEARS, SENIOR_YEARS = 2.0, 5.0


def seniority(title: str, total_yoe: float) -> str:
    low = title.lower()
    for level, words in SENIORITY_BY_TITLE:
        if any(w in low for w in words):
            return level
    if total_yoe < MID_YEARS:
        return "junior"
    return "mid" if total_yoe < SENIOR_YEARS else "senior"


def _latest(experiences: list[Experience], as_of: date) -> Experience | None:
    """The job that ends last (a current job ends 'now'); the first listed if none is dated."""
    dated = [(span[1], e) for e in experiences for span in intervals([e], as_of)]
    if dated:
        return max(dated, key=lambda t: t[0])[1]
    return experiences[0] if experiences else None


def _skill_evidence(
    vocab: SkillVocabulary,
    skills_text: str,
    summary_text: str,
    experiences: list[Experience],
    as_of: date,
) -> list[SkillEvidence]:
    claimed = dict.fromkeys(vocab.split_list(skills_text) + vocab.find(summary_text))
    used_in: dict[str, list[Experience]] = {}
    for exp in experiences:
        text = "\n".join([exp.title, exp.description, *exp.bullets])
        for skill in vocab.find(text):
            used_in.setdefault(skill, []).append(exp)
    out = []
    for skill in dict.fromkeys([*claimed, *used_in]):
        jobs = used_in.get(skill, [])
        years = round(months_covered([s for j in jobs for s in intervals([j], as_of)]) / 12, 1)
        out.append(
            SkillEvidence(
                skill=skill, claimed=skill in claimed, used=bool(jobs), years=years or None
            )
        )
    return out


def build_profile(
    markdown: str,
    *,
    as_of: date,
    gap_months: int = DEFAULT_GAP_MONTHS,
    llm: LLMProvider | None = None,
    vocab: SkillVocabulary | None = None,
    matcher: SectionMatcher | None = None,
) -> CandidateProfile:
    vocab = vocab or default_vocabulary()
    matcher = matcher or load_matcher()
    sections = split_sections(markdown, matcher)
    warnings: list[str] = []

    top = section_lines(sections, "header") + section_lines(sections, "contact")
    area = "\n".join(top) or markdown
    contact = Contact(
        email=find_email(area) or find_email(markdown),
        phone=find_phone(area) or find_phone(markdown),
        linkedin=find_linkedin(markdown),
        github=find_github(markdown),
        location=find_location(top),
    )
    name = find_name(markdown.splitlines(), matcher)

    experiences = parse_experiences(section_lines(sections, "experience"))
    if llm is not None:
        llm_extract.enrich(llm, experiences)
    education = parse_education(section_lines(sections, "education"))
    certifications = parse_certifications(section_lines(sections, "certifications"))
    skills = _skill_evidence(
        vocab,
        "\n".join(section_lines(sections, "skills")),
        "\n".join(section_lines(sections, "summary")),
        experiences,
        as_of,
    )
    total_yoe, gaps = compute_yoe(experiences, as_of, gap_months)
    latest = _latest(experiences, as_of)
    headline = latest.title if latest else ""

    for label, present in (
        ("experience", experiences), ("education", education), ("skills", skills),
    ):
        if not present:
            warnings.append(f"no {label} found")
    warnings += [f"unknown section: {s.heading}" for s in sections if s.label == "unknown"]
    if undated := sum(1 for e in experiences if not e.start):
        warnings.append(f"{undated} experience entries have no start date")
    if low := sum(1 for e in experiences if e.confidence != "high"):
        warnings.append(f"{low} experience entries have an uncertain title/company split")
    share = known_token_share(sections)
    if share < MIN_KNOWN_TOKEN_SHARE:
        warnings.append(f"only {share:.0%} of the text is in recognised sections")

    return CandidateProfile(
        name=name,
        email_hash=email_hash(contact.email),
        contact=contact,
        headline=headline,
        seniority=seniority(headline, total_yoe),
        total_yoe=total_yoe,
        gaps=gaps,
        experiences=experiences,
        education=education,
        skills=skills,
        certifications=certifications,
        sections=[SectionInfo(label=s.label, heading=s.heading, method=s.method, text=s.text)
                  for s in sections],
        warnings=warnings,
    )
