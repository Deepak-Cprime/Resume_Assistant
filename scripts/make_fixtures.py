"""Generate the synthetic resume fixtures into a folder.

Usage: python scripts/make_fixtures.py [output_dir]      (default: data/fixtures)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.builder import write_all  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/fixtures")
    for spec_id, path in write_all(out).items():
        print(f"{spec_id:28s} -> {path}")


if __name__ == "__main__":
    main()
