"""End to end on the synthetic resumes: real parsers -> sections -> fields, scored against the
ground truth in tests/fixtures/specs. Thresholds are the Phase 1 gates (plan section 7.5).
Uses rules only (no LLM), so it is deterministic and runs by default."""

from src.structure.sections import known_labels, known_token_share, split_sections
from src.structure.text import normalize_key

# `parsed` (cleaned Markdown) and `profiles` (CandidateProfile) per fixture come from conftest.
FIELD_GATE = 0.90
YOE_TOLERANCE = 0.5
MIN_SECTIONS = 4


def same(a: str, b: str) -> bool:
    return normalize_key(a) == normalize_key(b)


def score(specs, profiles, extract):
    """(accuracy, misses) over every expected value; misses name the fixture and field."""
    hits = total = 0
    misses = []
    for spec_id, spec in specs.items():
        for label, expected, actual in extract(spec, profiles[spec_id]):
            total += 1
            if expected == actual:
                hits += 1
            else:
                misses.append(f"{spec_id}: {label}: expected {expected!r}, got {actual!r}")
    return hits / total, misses


def jobs(spec, profile):
    got = profile.experiences
    for i, job in enumerate(spec["experience"]):
        yield i, job, (got[i] if i < len(got) else None)


def test_company_and_title(specs, profiles):
    def extract(spec, profile):
        for i, job, exp in jobs(spec, profile):
            yield f"company[{i}]", True, bool(exp and same(job["company"], exp.company))
            yield f"title[{i}]", True, bool(exp and same(job["title"], exp.title))

    accuracy, misses = score(specs, profiles, extract)
    assert accuracy >= FIELD_GATE, "\n".join(misses)


def test_dates(specs, profiles):
    def extract(spec, profile):
        for i, job, exp in jobs(spec, profile):
            yield f"start[{i}]", job["start"], exp.start if exp else None
            expected_end = None if job["end"] == "present" else job["end"]
            yield f"end[{i}]", expected_end, exp.end if exp else "missing"
            yield f"current[{i}]", job["end"] == "present", bool(exp and exp.is_current)

    accuracy, misses = score(specs, profiles, extract)
    assert accuracy >= FIELD_GATE, "\n".join(misses)


def test_degree_and_school(specs, profiles):
    def extract(spec, profile):
        edu = profile.education[0] if profile.education else None
        want = spec["education"][0]
        yield "degree", True, bool(edu and same(want["degree"], edu.degree))
        yield "school", True, bool(edu and same(want["school"], edu.school))
        yield "year", want["year"], edu.year if edu else None

    accuracy, misses = score(specs, profiles, extract)
    assert accuracy >= FIELD_GATE, "\n".join(misses)


def test_contact_and_name(specs, profiles):
    def extract(spec, profile):
        yield "name", True, same(spec["name"], profile.name)
        yield "email", spec["email"], profile.contact.email
        yield "phone", spec["phone"], profile.contact.phone

    accuracy, misses = score(specs, profiles, extract)
    assert accuracy >= FIELD_GATE, "\n".join(misses)


def test_certifications_and_skills(specs, profiles):
    def extract(spec, profile):
        names = [c.name for c in profile.certifications]
        for cert in spec["certifications"]:
            yield f"cert {cert}", True, any(same(cert, n) for n in names)

    accuracy, misses = score(specs, profiles, extract)
    assert accuracy >= FIELD_GATE, "\n".join(misses)

    found = {s.skill.lower() for p in profiles.values() for s in p.skills}
    assert {"python", "kubernetes", "terraform", "react"} <= found


def test_years_of_experience(specs, profiles):
    bad = [
        f"{k}: expected {s['expected']['total_yoe']}, got {profiles[k].total_yoe}"
        for k, s in specs.items()
        if abs(profiles[k].total_yoe - s["expected"]["total_yoe"]) > YOE_TOLERANCE
    ]
    assert 1 - len(bad) / len(specs) >= FIELD_GATE, "\n".join(bad)


def test_overlap_is_not_double_counted(profiles):
    assert profiles["single_col_overlap"].total_yoe < 4.0  # the sum of the two jobs is 5.0


def test_career_gap_is_found(profiles):
    gaps = profiles["single_col_gap"].gaps
    assert [(g.start, g.end, g.months) for g in gaps] == [("2018-03", "2019-09", 18)]


def test_sections_found(specs, parsed):
    bad = []
    shares = []
    for spec_id, md in parsed.items():
        sections = split_sections(md)
        shares.append(known_token_share(sections))
        if known_labels(sections) < MIN_SECTIONS:
            bad.append(f"{spec_id}: only {known_labels(sections)} known sections")
    assert not bad, "\n".join(bad)
    assert sum(shares) / len(shares) >= FIELD_GATE, shares


def test_expected_section_labels_are_present(specs, parsed):
    missing = []
    for spec_id, spec in specs.items():
        found = {s.label for s in split_sections(parsed[spec_id])}
        for label in spec["expected"]["sections"]:
            if label not in found:
                missing.append(f"{spec_id}: no {label} section (found {sorted(found)})")
    assert not missing, "\n".join(missing)
