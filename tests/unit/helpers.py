def norm(text: str) -> str:
    return " ".join(text.split())


def check_reading_order(markdown: str, spec: dict) -> None:
    """Experience entries appear in order, their bullets stay together, and nothing from the
    sidebar (contact, certifications, skills not used in bullets) lands in between."""
    text = norm(markdown)
    jobs = spec["experience"]
    firsts = [text.find(norm(j["bullets"][0])[:40]) for j in jobs]
    lasts = [text.find(norm(j["bullets"][-1])[:40]) for j in jobs]
    assert -1 not in firsts + lasts, "experience bullets missing from output"
    assert firsts == sorted(firsts), "jobs out of order"
    span = (min(firsts), max(lasts))

    bullets_text = " ".join(b for j in jobs for b in j["bullets"]).lower()
    unique_skills = [s for s in spec["skills"] if s.lower() not in bullets_text]
    assert unique_skills, "fixture needs a skill that is absent from the bullets"
    markers = [spec["email"], spec["phone"], *spec["certifications"], *unique_skills]
    for marker in markers:
        pos = text.find(marker)
        assert pos != -1, f"{marker!r} missing from output"
        assert not span[0] < pos < span[1], f"{marker!r} interleaved with experience text"
