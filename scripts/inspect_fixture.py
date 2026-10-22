"""Show how fixture resumes are parsed: cleaned Markdown, detected sections, extracted profile.

Usage: python scripts/inspect_fixture.py single_col_backend scanned_devops
       python scripts/inspect_fixture.py            (lists the available fixtures)
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.metadata.profile import build_profile  # noqa: E402
from src.providers.parsers import router  # noqa: E402
from src.structure.sections import split_sections  # noqa: E402
from tests.fixtures.builder import load_specs, write_all  # noqa: E402


def show(spec_id: str, path: Path, settings) -> None:
    doc_type = "docx" if path.suffix == ".docx" else "pdf"
    outcome = router.parse_resume(path, doc_type, settings)
    print(f"\n{'=' * 20} {spec_id} ({outcome.parsed.parser}) {'=' * 20}")
    print("--- markdown (repr, one line each) ---")
    for i, line in enumerate(outcome.markdown.splitlines()):
        print(f"{i:3d} {line!r}")
    print("--- sections ---")
    for s in split_sections(outcome.markdown):
        print(f"{s.label:15s} {s.method:10s} {s.heading!r:28s} {len(s.text.split()):4d} words")
    p = build_profile(outcome.markdown, as_of=date(2025, 6, 1))
    print("--- profile ---")
    print("name:", p.name, "| email:", p.contact.email, "| phone:", p.contact.phone)
    for e in p.experiences:
        print(f"job: {e.title!r} @ {e.company!r} {e.start}..{e.end} current={e.is_current} "
              f"[{e.confidence}]")
    for e in p.education:
        print(f"edu: {e.degree!r} | {e.school!r} | {e.year}")
    print("certs:", [c.name for c in p.certifications])
    print("skills:", [s.skill for s in p.skills])
    print("yoe:", p.total_yoe, "gaps:", [(g.start, g.end) for g in p.gaps])
    print("warnings:", p.warnings)


def main() -> None:
    specs = load_specs()
    wanted = sys.argv[1:]
    if not wanted:
        print("fixtures:", ", ".join(specs))
        return
    settings = load_settings()
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_all(Path(tmp))
        for spec_id in wanted:
            show(spec_id, paths[spec_id], settings)


if __name__ == "__main__":
    main()
