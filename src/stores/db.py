"""SQLite schema and repository functions. All SQL lives here."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ROLE_ID = 1

# Pipeline order; a document moves forward one stage at a time.
PIPELINE_STATUSES = ("uploaded", "parsed", "structured", "extracted", "chunked", "indexed")
FLAG_STATUSES = ("failed", "needs_review")
ALL_STATUSES = PIPELINE_STATUSES + FLAG_STATUSES

SCHEMA = """
CREATE TABLE IF NOT EXISTS roles (
    role_id       INTEGER PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    jd_text       TEXT NOT NULL DEFAULT '',
    chunk_budgets TEXT,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name  TEXT,
    email_hash    TEXT,
    headline      TEXT,
    total_yoe     REAL,
    seniority     TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    candidate_id    INTEGER REFERENCES candidates(candidate_id) ON DELETE SET NULL,
    filename        TEXT NOT NULL,
    doc_type        TEXT,
    raw_path        TEXT,
    parser          TEXT,
    parser_version  TEXT,
    status          TEXT NOT NULL,
    error           TEXT,
    markdown_path   TEXT,
    chunk_fp        TEXT,               -- budgets fingerprint its chunks were built with
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiences (
    exp_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  INTEGER NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    title         TEXT,
    company       TEXT,
    start         TEXT,
    end           TEXT,
    chunk_id      INTEGER
);
CREATE TABLE IF NOT EXISTS skills (
    candidate_id  INTEGER NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    skill         TEXT NOT NULL,
    evidence      TEXT,
    years         REAL,
    chunk_ids     TEXT,
    PRIMARY KEY (candidate_id, skill)
);
CREATE TABLE IF NOT EXISTS certifications (
    candidate_id  INTEGER NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    year          INTEGER,
    PRIMARY KEY (candidate_id, name)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY,          -- 60-bit hash of chunk_key (FAISS needs ints)
    chunk_key     TEXT NOT NULL UNIQUE,         -- readable, deterministic: c12-exp-1-p1
    candidate_id  INTEGER NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    role_id       INTEGER NOT NULL REFERENCES roles(role_id),
    doc_id        TEXT NOT NULL,
    level         INTEGER NOT NULL,             -- 0 profile, 1 section, 2 entry
    parent_id     INTEGER,
    section       TEXT,
    entry_title   TEXT,
    start         TEXT,
    end           TEXT,
    skills        TEXT,                         -- JSON list of skill tags
    text          TEXT NOT NULL,                -- shown to the user and cited
    embed_text    TEXT NOT NULL,                -- prefix + masked text + tags (embedded, BM25)
    embed         INTEGER NOT NULL DEFAULT 1,   -- 0 = stored parent, not embedded
    n_tokens      INTEGER,
    page          INTEGER,
    content_hash  TEXT
);
CREATE TABLE IF NOT EXISTS index_manifest (
    index_version INTEGER PRIMARY KEY AUTOINCREMENT,
    emb_model     TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    chunker_cfg   TEXT,
    n_vectors     INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_candidate ON chunks(candidate_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class InvalidTransition(ValueError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database, creating the file, schema and the single role row if needed."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO roles (role_id, updated_at) VALUES (?, ?)",
        (DEFAULT_ROLE_ID, _now()),
    )
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


# --- roles -----------------------------------------------------------------------------


def get_role(conn: sqlite3.Connection, role_id: int = DEFAULT_ROLE_ID) -> sqlite3.Row:
    return conn.execute("SELECT * FROM roles WHERE role_id = ?", (role_id,)).fetchone()


def update_role(
    conn: sqlite3.Connection, *, title: str, jd_text: str, role_id: int = DEFAULT_ROLE_ID
) -> None:
    """Editing the JD never touches documents, chunks or indexes."""
    with transaction(conn):
        conn.execute(
            "UPDATE roles SET title = ?, jd_text = ?, updated_at = ? WHERE role_id = ?",
            (title, jd_text, _now(), role_id),
        )


def set_chunk_budgets(
    conn: sqlite3.Connection, budgets: dict, role_id: int = DEFAULT_ROLE_ID
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE roles SET chunk_budgets = ?, updated_at = ? WHERE role_id = ?",
            (json.dumps(budgets, sort_keys=True), _now(), role_id),
        )


# --- documents -------------------------------------------------------------------------


def check_transition(old: str, new: str) -> None:
    """Valid: any -> failed/needs_review; failed/needs_review -> uploaded (retry);
    next pipeline stage; or back to an earlier stage (re-run after a config change)."""
    if old not in ALL_STATUSES or new not in ALL_STATUSES:
        raise InvalidTransition(f"unknown status in {old!r} -> {new!r}")
    if new in FLAG_STATUSES:
        return
    if old in FLAG_STATUSES:
        if new == "uploaded":
            return
        raise InvalidTransition(f"{old} -> {new}: retry must restart from 'uploaded'")
    i, j = PIPELINE_STATUSES.index(old), PIPELINE_STATUSES.index(new)
    if j == i + 1 or j < i:
        return
    raise InvalidTransition(f"{old} -> {new}: stages cannot be skipped")


def insert_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    filename: str,
    doc_type: str | None,
    raw_path: str | None,
    status: str = "uploaded",
    error: str | None = None,
) -> bool:
    """Insert once per content hash. Returns False (and changes nothing) if it exists."""
    now = _now()
    with transaction(conn):
        cur = conn.execute(
            "INSERT OR IGNORE INTO documents "
            "(doc_id, filename, doc_type, raw_path, status, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, filename, doc_type, raw_path, status, error, now, now),
        )
    return cur.rowcount == 1


def get_document(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()


def list_documents(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    if status is None:
        return conn.execute("SELECT * FROM documents ORDER BY created_at, doc_id").fetchall()
    return conn.execute(
        "SELECT * FROM documents WHERE status = ? ORDER BY created_at, doc_id", (status,)
    ).fetchall()


def set_document_status(
    conn: sqlite3.Connection,
    doc_id: str,
    status: str,
    error: str | None = None,
    **fields: str | int | None,
) -> None:
    """Validate the transition and update the row. Extra `fields` may be parser,
    parser_version, markdown_path or candidate_id."""
    allowed = {"parser", "parser_version", "markdown_path", "candidate_id", "chunk_fp"}
    if not set(fields) <= allowed:
        raise ValueError(f"unexpected fields: {sorted(set(fields) - allowed)}")
    row = get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    check_transition(row["status"], status)
    sets = ["status = ?", "error = ?", "updated_at = ?"] + [f"{k} = ?" for k in fields]
    with transaction(conn):
        conn.execute(
            f"UPDATE documents SET {', '.join(sets)} WHERE doc_id = ?",
            (status, error, _now(), *fields.values(), doc_id),
        )


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


# --- candidates and their extracted metadata -------------------------------------------


def upsert_candidate_profile(
    conn: sqlite3.Connection,
    *,
    display_name: str,
    email_hash: str,
    headline: str,
    total_yoe: float,
    seniority: str,
    experiences: list[dict],
    skills: list[dict],
    certifications: list[dict],
) -> int:
    """Create the candidate (or, when the email hash matches one, refresh it) and replace its
    experiences, skills and certifications in one transaction. Returns the candidate id."""
    with transaction(conn):
        row = (
            conn.execute(
                "SELECT candidate_id FROM candidates WHERE email_hash = ?", (email_hash,)
            ).fetchone()
            if email_hash
            else None
        )
        if row is None:
            cur = conn.execute(
                "INSERT INTO candidates (display_name, email_hash, headline, total_yoe, seniority)"
                " VALUES (?, ?, ?, ?, ?)",
                (display_name, email_hash or None, headline, total_yoe, seniority),
            )
            candidate_id = int(cur.lastrowid)
        else:
            candidate_id = int(row["candidate_id"])
            conn.execute(
                "UPDATE candidates SET display_name = ?, headline = ?, total_yoe = ?, "
                "seniority = ? WHERE candidate_id = ?",
                (display_name, headline, total_yoe, seniority, candidate_id),
            )
            for table in ("experiences", "skills", "certifications"):
                conn.execute(f"DELETE FROM {table} WHERE candidate_id = ?", (candidate_id,))
        conn.executemany(
            "INSERT INTO experiences (candidate_id, title, company, start, end) "
            "VALUES (?, ?, ?, ?, ?)",
            [(candidate_id, e["title"], e["company"], e["start"], e["end"]) for e in experiences],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO skills (candidate_id, skill, evidence, years) "
            "VALUES (?, ?, ?, ?)",
            [(candidate_id, s["skill"], s["evidence"], s["years"]) for s in skills],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO certifications (candidate_id, name, year) VALUES (?, ?, ?)",
            [(candidate_id, c["name"], c["year"]) for c in certifications],
        )
    return candidate_id


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()


def list_experiences(conn: sqlite3.Connection, candidate_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM experiences WHERE candidate_id = ? ORDER BY exp_id", (candidate_id,)
    ).fetchall()


def list_skills(conn: sqlite3.Connection, candidate_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM skills WHERE candidate_id = ? ORDER BY skill", (candidate_id,)
    ).fetchall()


def list_certifications(conn: sqlite3.Connection, candidate_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM certifications WHERE candidate_id = ? ORDER BY name", (candidate_id,)
    ).fetchall()


# --- chunks and budgets ----------------------------------------------------------------

_CHUNK_COLUMNS = (
    "chunk_id", "chunk_key", "candidate_id", "role_id", "doc_id", "level", "parent_id",
    "section", "entry_title", "start", "end", "skills", "text", "embed_text", "embed",
    "n_tokens", "page", "content_hash",
)


def replace_chunks(
    conn: sqlite3.Connection, candidate_id: int, chunks: list[dict], role_id: int = DEFAULT_ROLE_ID
) -> None:
    """Swap a candidate's chunks for a new set in one transaction (re-running never duplicates)."""
    placeholders = ", ".join("?" for _ in _CHUNK_COLUMNS)
    rows = [tuple({**c, "role_id": role_id}[k] for k in _CHUNK_COLUMNS) for c in chunks]
    with transaction(conn):
        conn.execute("DELETE FROM chunks WHERE candidate_id = ?", (candidate_id,))
        conn.executemany(
            f"INSERT INTO chunks ({', '.join(_CHUNK_COLUMNS)}) VALUES ({placeholders})", rows
        )


def list_chunks(
    conn: sqlite3.Connection, *, candidate_id: int | None = None, embedded_only: bool = False
) -> list[sqlite3.Row]:
    where, args = [], []
    if candidate_id is not None:
        where.append("candidate_id = ?")
        args.append(candidate_id)
    if embedded_only:
        where.append("embed = 1")
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    return conn.execute(f"SELECT * FROM chunks{clause} ORDER BY chunk_id", args).fetchall()


def count_chunks(conn: sqlite3.Connection, *, embedded_only: bool = False) -> int:
    sql = "SELECT COUNT(*) FROM chunks" + (" WHERE embed = 1" if embedded_only else "")
    return int(conn.execute(sql).fetchone()[0])


def get_chunk_budgets(conn: sqlite3.Connection, role_id: int = DEFAULT_ROLE_ID) -> dict | None:
    raw = get_role(conn, role_id)["chunk_budgets"]
    return json.loads(raw) if raw else None


# --- index manifests -------------------------------------------------------------------


def add_index_manifest(
    conn: sqlite3.Connection, *, emb_model: str, dim: int, chunker_cfg: dict, n_vectors: int
) -> int:
    """Record one index build (history is kept; the newest row per model is current)."""
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO index_manifest (emb_model, dim, chunker_cfg, n_vectors, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (emb_model, dim, json.dumps(chunker_cfg, sort_keys=True), n_vectors, _now()),
        )
    return int(cur.lastrowid)


def current_index_manifests(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """The newest manifest row of every embedding model."""
    return conn.execute(
        "SELECT * FROM index_manifest WHERE index_version IN "
        "(SELECT MAX(index_version) FROM index_manifest GROUP BY emb_model) ORDER BY emb_model"
    ).fetchall()


# --- pipeline queries and deletion -----------------------------------------------------

PENDING_STATUSES = ("uploaded", "parsed", "structured", "extracted")  # work left to do


def list_pending_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    marks = ", ".join("?" for _ in PENDING_STATUSES)
    return conn.execute(
        f"SELECT * FROM documents WHERE status IN ({marks}) ORDER BY created_at, doc_id",
        PENDING_STATUSES,
    ).fetchall()


def chunk_owner_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Documents whose chunks are the live ones (an older upload of the same candidate is
    replaced and owns none)."""
    return conn.execute(
        "SELECT * FROM documents WHERE doc_id IN (SELECT DISTINCT doc_id FROM chunks) "
        "ORDER BY created_at, doc_id"
    ).fetchall()


def documents_of_candidate(conn: sqlite3.Connection, candidate_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE candidate_id = ? ORDER BY created_at, doc_id",
        (candidate_id,),
    ).fetchall()


def candidate_chunk_ids(conn: sqlite3.Connection, candidate_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE candidate_id = ? ORDER BY chunk_id", (candidate_id,)
    ).fetchall()
    return [int(r["chunk_id"]) for r in rows]


def delete_candidate_rows(conn: sqlite3.Connection, candidate_id: int) -> None:
    """Remove a candidate with every document, experience, skill, certification and chunk."""
    with transaction(conn):
        conn.execute("DELETE FROM documents WHERE candidate_id = ?", (candidate_id,))
        conn.execute("DELETE FROM candidates WHERE candidate_id = ?", (candidate_id,))  # cascades


def delete_document_row(conn: sqlite3.Connection, doc_id: str) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
