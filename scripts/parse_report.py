"""Run the pipeline on a folder of resumes and write a parse report with the Phase 1 gates.

Usage: python scripts/parse_report.py                      synthetic fixtures (with ground truth)
       python scripts/parse_report.py --dir local_resumes  your own PDFs/DOCX
       python scripts/parse_report.py --embed              also build the real vector + BM25 index

Works in a throwaway data folder, so the real `data/` is never touched. The report is written
to data/reports/parse_report.md (override with --out).
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.pipeline import report, uploads  # noqa: E402
from src.pipeline import run as pipeline  # noqa: E402
from src.stores import db  # noqa: E402


def collect(directory: Path | None, scratch: Path):
    """(files, expected YOE by file stem). Fixtures are generated; a folder is read as is."""
    if directory is not None:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".pdf", ".docx"))
        return files, {}
    from tests.fixtures.builder import load_specs, write_all

    specs = load_specs()
    paths = write_all(scratch / "fixtures")
    return sorted(paths.values()), {k: v["expected"]["total_yoe"] for k, v in specs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, help="folder of resumes (default: synthetic fixtures)")
    ap.add_argument("--embed", action="store_true", help="build the real embedding + BM25 index")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reports" / "parse_report.md")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        os.environ["RAG_DATA_DIR"] = str(scratch / "data")
        settings = load_settings()
        if args.dir is None:  # the fixtures' ground truth treats "present" as this date
            from tests.conftest import FIXTURE_AS_OF

            settings.metadata.as_of = FIXTURE_AS_OF
        conn = db.connect(settings.paths.db)
        files, expected = collect(args.dir, scratch)
        results = uploads.ingest_files(conn, settings, [(f.name, f.read_bytes()) for f in files])

        seconds: dict[str, float] = {}
        for r in results:
            if not r.doc_id:
                continue
            start = time.perf_counter()
            pipeline.process_document(conn, settings, r.doc_id)
            seconds[r.doc_id] = time.perf_counter() - start
            print(f"{r.filename}: {seconds[r.doc_id]:.1f}s")
        pipeline.run_pipeline(conn, settings, build_indexes=args.embed)

        reports = report.document_reports(conn, settings, seconds, expected)
        gates = report.compute_gates(reports, conn, settings)
        title = f"Parse report: {args.dir or 'synthetic fixtures'}"
        text = report.render_markdown(reports, gates, title)
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"\n{text}\nWritten to {args.out}")
    return 0 if not any(g.failed for g in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
