# Phase 1 Development Plan: Resume Indexing

| | |
|---|---|
| **Status** | Draft, approved scope |
| **Phase** | 1 of 3 (Indexing). Phase 2 = retrieval + generation, Phase 3 = quality |
| **Source of truth** | `CLAUDE.md`. If this document and `CLAUDE.md` disagree, `CLAUDE.md` wins; update this file to match. |
| **Background reading** | `docs/guides/phase1_indexing_guide.html` (why each method was chosen) |
| **Architecture** | `docs/rag_resume_assistant_architecture.excalidraw` |

---

## 1. Goal

Turn a batch of resumes submitted for one job role into three persisted, searchable stores
(dense vectors, BM25 keyword index, metadata DB) that Phase 2 can query, with a Streamlit UI to
upload, inspect and index resumes.

### In scope

- One role with an editable job description.
- PDF (digital and scanned) and DOCX resumes.
- Parsing, section detection, metadata extraction, adaptive chunking, embedding with two models,
  BM25, SQLite + FAISS persistence.
- Streamlit UI for the indexing workflow.
- A full testing phase (section 7).

### Out of scope (later phases)

- Asking questions, retrieval, reranking, LLM answers (Phase 2).
- Retrieval and generation metrics, model comparison reports (Phase 3).
- Multiple roles, authentication, deployment, a REST API.

---

## 2. Scope decisions

| Decision | Choice | Reason |
|---|---|---|
| Roles | One role; `roles` table still exists with one row | Multi-role later without schema change |
| Job description | Editable in the UI, stored in `roles.jd_text` | Used only at query time, so editing never re-indexes |
| UI | Streamlit calling `src/` directly | Pure Python, fast to build; FastAPI can be added later on the same `src/` |
| Backend server | None (no Flask/FastAPI) | Streamlit is the only process for the assignment |
| Persistence | SQLite + FAISS files + bm25s files under `data/` | Survives restarts; no server to run |
| LLM / embeddings | Ollama + sentence-transformers | Open source; swappable via provider interfaces |
| Execution | Claude writes code; the user runs installs, scripts, tests and the app | Per `CLAUDE.md` |

---

## 3. Architecture summary

```
Streamlit pages ──► src/pipeline ──► src/ingestion ─► src/providers/parsers
                         │          ─► src/structure
                         │          ─► src/metadata   ─► src/providers/llm (Ollama)
                         │          ─► src/chunking
                         │          ─► src/providers/embedders ─► src/providers/vectorstores (FAISS)
                         │          ─► src/providers/keyword (bm25s)
                         └────────────► src/stores (SQLite, files)
```

- Pages call `src/pipeline` and `src/stores` only. No business logic in `app/`.
- Every external dependency sits behind a Protocol in `src/providers/base.py` and is created by a
  factory from `configs/config.yaml`.

### Project layout

```
app/
  Home.py               role + editable JD, corpus stats
  pages/1_Upload.py     upload PDFs/DOCX, per-document status, retry failed
  pages/2_Inspect.py    per resume: raw → markdown → sections → metadata → chunks
  pages/3_Index.py      build/rebuild per embedding model, manifest, stats, delete candidate
src/
  config.py
  providers/  base.py, factory.py, parsers/, embedders/, llm/, vectorstores/, keyword/
  ingestion/  intake.py, cleaning.py, validation.py
  structure/  sections.py, entries.py, dates.py, text.py
  metadata/   schema.py, regex_extract.py, llm_extract.py, skills.py, pii.py, yoe.py, profile.py
  chunking/   chunker.py, budgets.py, enrich.py, tokens.py
  stores/     db.py, files.py
  pipeline/   run.py, delete.py, manifest.py, parse_stage.py, structure_stage.py,
              extract_stage.py, chunk_stage.py, index_stage.py, keyword_stage.py (one per stage),
              overview.py (read-only views for the UI), uploads.py (batch intake)
configs/      config.yaml, sections_resume.yaml, skills_aliases.yaml
data/         raw/, markdown/, profiles/, chunks/, indexes/, app.db      (gitignored)
tests/        unit/, integration/, ui/, fixtures/, conftest.py
eval/         golden_questions.yaml
scripts/      make_fixtures.py, parse_report.py
```

### Data model (SQLite)

| Table | Key | Main columns |
|---|---|---|
| roles | role_id | title, jd_text, chunk_budgets (JSON), updated_at |
| candidates | candidate_id | display_name, email_hash, headline, total_yoe, seniority |
| documents | doc_id (sha256) | candidate_id, filename, doc_type, parser, parser_version, status, error, markdown_path |
| experiences | exp_id | candidate_id, title, company, start, end, chunk_id |
| skills | (candidate_id, skill) | evidence (claimed/used), years, chunk_ids |
| certifications | (candidate_id, name) | year |
| chunks | chunk_id | candidate_id, role_id, doc_id, level, parent_id, section, text, embed_text, n_tokens, page, content_hash |
| index_manifest | index_version | emb_model, dim, chunker_cfg, n_vectors, created_at |

Document status flow: `uploaded → parsed → structured → extracted → chunked → indexed`, or
`failed` / `needs_review` with an error message.

---

## 4. Milestones

Build in order. Each milestone ends with its tests written and passing (section 7.4).

| # | Milestone | Deliverables | Done when |
|---|---|---|---|
| M0 | Scaffolding | Layout, `requirements.txt`, `requirements-dev.txt`, `config.yaml`, `config.py`, Protocols + factory, logging, `pytest` + `ruff` config, fixture generator script | `import src` works; config loads; `pytest` runs (empty) |
| M1 | Storage + intake | SQLite schema + repository functions; validate (magic bytes, size, pages, encryption); sha256; raw file store; status table | Re-uploading a file creates nothing new |
| M2 | Parsing | pymupdf4llm parser (column-aware for sidebar pages, PyMuPDF only), python-docx parser, optional Docling parser (off by default); router (chars/page, column detection, DOCX, scanned); validation; cleaning; Markdown saved | Fixture resumes parse in correct reading order |
| M3 | Structure | Section detection (synonyms + heuristics + fuzzy), entry splitting, date normalisation | ≥ 90% of tokens in known sections on fixtures |
| M4 | Metadata | Regex contact fields; YOE as interval union; gaps; skill aliases (claimed vs used); Ollama JSON-schema extraction per entry with grounding check; PII masking | Field accuracy ≥ 90% on labelled fixtures |
| M5 | Chunking | Section + entry chunker; adaptive per-section budgets stored on the role; structural prefix; profile chunk; parent links; deterministic chunk ids | 0 chunks over model limit; every experience chunk has company + dates |
| M6 | Embeddings + vectors | sentence-transformers embedder (bge-small); Ollama embedder (nomic, optional); FAISS `IndexIDMap(IndexFlatIP)` per model, persisted; manifest | Two model indexes side by side; counts match DB |
| M7 | Keyword index | bm25s with tech-aware tokenizer; persisted | C#, C++, .NET stay distinct tokens |
| M8 | Orchestration | `pipeline.run` with per-stage status, content-hash skip, re-chunk on budget change, `delete_candidate` across all stores | Idempotency and delete tests pass |
| M9 | Streamlit UI | Four pages; `st.cache_resource` for heavy objects; progress per document | UI smoke tests pass; manual flow works |
| M10 | Testing phase + exit | Full test suite run, parse report, chunk stats, golden questions, exit review (section 7) | All exit criteria in section 8 met |

---

## 5. Configuration

```yaml
# configs/config.yaml
use_case: resume
paths:
  data_dir: data
  db: data/app.db
ingestion:
  max_file_mb: 10
  max_pages: 10
parser:
  default: pymupdf4llm      # digital PDFs, incl. two-column (column-aware, no ML)
  docx: python_docx
  fallback: null            # "docling" = heavy retry for unusable results (needs torch)
  ocr: tesseract
  vlm_fallback: {enabled: false, model: qwen2.5vl:7b}
  min_chars_per_page: 50
llm:
  provider: ollama
  model: qwen2.5:7b
  temperature: 0
embedding:
  models:
    - {provider: sentence_transformers, name: BAAI/bge-small-en-v1.5}
    # optional challenger, runs in Ollama: - {provider: ollama, name: nomic-embed-text}
  default: BAAI/bge-small-en-v1.5
chunking:
  strategy: section_entry
  budgets: {mode: adaptive_per_section, p_low: 20, p_high: 90,
            min_clamp: [64, 128], max_clamp: [256, null], prefix_reserve: 32}
vector_store: {provider: faiss, index: flat_ip}
keyword: {provider: bm25s, tokenizer: tech_aware}
```

Secrets (paid API keys later) come from environment variables only.

---

## 6. Streamlit UI

| Page | Purpose | Key elements |
|---|---|---|
| Home | Role and JD | Role title, JD text area with Save, counts (resumes, candidates, chunks, failed) |
| Upload | Add resumes | Multi-file uploader, "Process" button, status table per document, retry for failed, "needs review" list |
| Inspect | See each stage for one resume | Candidate picker; tabs: Markdown, Sections (with canonical labels), Metadata (JSON + YOE timeline), Chunks (prefix, tokens, parent) |
| Index | Build and manage indexes | Model picker, Build / Rebuild, manifest table, chunk-size histogram, delete candidate |

Rules: heavy objects via `st.cache_resource`; long work inside `st.status` with per-document
progress; every action goes through `src/pipeline` so it is testable without the UI.

---

## 7. Testing phase

Testing runs alongside every milestone (tests written with the code) and as a dedicated phase at
M10 before Phase 1 is closed.

### 7.1 Test data

| Set | Contents | Source |
|---|---|---|
| Synthetic fixtures | 15–20 fictional resumes: single column, two-column, sidebar, tables, scanned (rasterised PDF), DOCX, one non-English, one with no headings, one with overlapping jobs, one with a career gap, one duplicate with different bytes | `scripts/make_fixtures.py` generates them from YAML ground truth |
| Ground truth | Per fixture YAML: expected sections, experiences (title, company, start, end), skills, certifications, total YOE | Written by hand alongside the generator |
| Edge-case files | Empty PDF, encrypted PDF, 50-page PDF, corrupt file, `.exe` renamed to `.pdf`, image-only page | Generated or hand-made in `tests/fixtures/bad/` |
| Real resumes (optional) | Only with consent; never committed | Local folder in `.gitignore` |

Fictional data only in the repo: names, emails (`@example.com`) and phone numbers must not
belong to real people.

### 7.2 Test levels

| Level | What | Tools | Runs |
|---|---|---|---|
| Unit | Pure functions: hashing, validation, cleaning, section mapping, entry split, date parsing, YOE union, skill aliases, tokenizer, budgets, chunker, prefix, chunk ids, PII masking | `pytest`, parametrised cases | Every change, < 10 s |
| Contract | Each provider implementation satisfies its Protocol (shapes, dims, normalisation, determinism) | `pytest` with a shared contract test per interface | Every change |
| Component | Parsers on fixture files; LLM extraction with a fake LLM; FAISS and bm25s save/load round-trip | `pytest`, `tmp_path` | Every change; Docling tests marked `slow` |
| Integration | Full `pipeline.run` over the fixture corpus into a temp `data/` dir; checks DB, vector and BM25 counts agree | `pytest -m integration` | Before each milestone is closed |
| Golden / snapshot | Parsed Markdown and chunk lists for 5 fixtures compared with saved snapshots, so parser or chunker changes are visible in diffs | Snapshot files in `tests/fixtures/snapshots/` | Every change; update snapshots deliberately |
| Accuracy | Extracted metadata vs ground truth YAML: per-field precision/recall, YOE error | `pytest -m accuracy` (needs Ollama) | M4, M10 |
| UI smoke | Each page loads, upload → process → inspect works on 2 fixtures | `streamlit.testing.v1.AppTest` | M9, M10 |
| Performance | Time per resume per stage on the fixture corpus; memory of indexes | `scripts/parse_report.py` | M10 |
| Manual QA | Visual check of 10 parsed resumes incl. 3 two-column; UI walkthrough | Checklist in 7.6 | M10 |

### 7.3 Test isolation

- **No network or Ollama in default runs.** A `FakeLLM` returns canned JSON per fixture; a
  `FakeEmbedder` returns deterministic hashed vectors with the right dimension. Real-model tests
  are marked `ollama` or `models` and skipped unless enabled.
- Every test uses `tmp_path` for `data/`; nothing touches the real `data/` directory.
- Pytest markers: `slow`, `integration`, `ollama`, `models`, `accuracy`, `ui`.
  Default run: `pytest -m "not slow and not ollama and not models"`.

### 7.4 Required tests per milestone

| Milestone | Must-have tests |
|---|---|
| M1 | Rejects bad magic bytes, oversize, encrypted; same bytes → same `doc_id`; second upload → no new rows; status transitions are valid |
| M2 | Router picks the expected parser for each fixture type; two-column fixture keeps skills out of experience text; scanned fixture goes to OCR; validation flags garbage; cleaning fixes ligatures, bullets, hyphenation, repeated footers |
| M3 | Every synonym maps to its canonical label; unknown heading flagged; name line not treated as a section; entries split on date-range headers; date formats (`Jan 2021`, `07/2018`, `2018`, `Present`) normalise correctly |
| M4 | YOE uses union, not sum (overlap fixture); gaps > 3 months detected; skill aliases (k8s → Kubernetes, golang → Go); claimed vs used; grounding rejects a company not in source text; PII absent from `embed_text` |
| M5 | Budgets computed from percentiles and clamped; no chunk exceeds `model_limit - prefix_reserve`; merges never cross entries or candidates; split parts repeat the entry header; chunk ids stable across runs; profile chunk present per candidate |
| M6 | Embedder contract (dim, L2-normalised, query prefix applied); FAISS save/load returns identical results; one index per model; manifest refuses a mismatched query model |
| M7 | Tokenizer keeps C++, C#, .NET, Node.js distinct; aliases applied; BM25 ranks the chunk with a rare skill first; save/load round-trip |
| M8 | Full run on fixtures: counts equal across DB, FAISS and BM25; re-run is a no-op; changed file re-indexes only its chunks; `delete_candidate` leaves zero rows/vectors/entries; a failing document does not stop the batch |
| M9 | AppTest: each page renders; JD save persists; upload + process of 2 fixtures reaches `indexed`; Inspect shows chunks |

### 7.5 Quality thresholds (Phase 1 gates)

| Metric | Threshold |
|---|---|
| Fixtures parsed without `needs_review` | ≥ 90% |
| Tokens in known sections | ≥ 90% |
| Resumes with ≥ 4 known sections | ≥ 95% |
| Metadata field accuracy (company, title, dates, degree) | ≥ 90% per field |
| YOE within ±0.5 years of ground truth | ≥ 90% of fixtures |
| Chunks under `min_tok` | < 5% (measured on realistic 5-10 year resumes in M10 QA; the synthetic fixtures have 2 short bullets per job, so the unit test only reports it) |
| Chunks over model limit | 0 |
| Experience chunks containing company and dates | 100% |
| Store count mismatch (DB vs FAISS vs BM25) | 0 |
| Unit + contract test coverage of `src/` | ≥ 80% lines |

### 7.6 Manual QA checklist (M10)

- [ ] Open 10 parsed resumes in Inspect (incl. 3 two-column, 1 scanned, 1 DOCX); reading order correct.
- [ ] Sections and canonical labels look right for each.
- [ ] Metadata JSON matches the resume; YOE timeline looks right.
- [ ] Chunks read as self-contained with their prefix.
- [ ] Nearest neighbours of 5 chunks (via a debug search) make sense.
- [ ] Edit the JD, restart the app: JD persists and nothing re-indexes.
- [ ] Delete a candidate: gone from every page and from index counts.
- [ ] Restart the app: indexes load from disk, no re-embedding.

### 7.7 Tooling

- `pytest`, `pytest-cov`, `ruff` (lint + format), `mypy` on `src/` (optional, non-blocking).
- `scripts/parse_report.py`: per document parser used, timings, sections found, chunk stats,
  warnings; writes `data/reports/parse_report.md`.
- CI (later, GitHub Actions): `ruff` + default `pytest` run on every push.

---

## 8. Phase 1 exit criteria

Phase 1 is closed when all of these hold:

1. All milestones M0–M10 done; their tests pass.
2. Every threshold in 7.5 met on the fixture corpus.
3. Manual QA checklist in 7.6 complete.
4. `eval/golden_questions.yaml` has 40+ questions with expected `chunk_id`s and candidate ids,
   drafted while inspecting fixtures.
5. Parse report generated and reviewed.
6. `CLAUDE.md` guide status updated; Phase 2 study guide is next.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Docling slow or heavy on CPU | Laptop struggles, long indexing | Off by default (`parser.fallback: null`); default path is PyMuPDF only; enable Docling only for scanned PDFs; tests marked `slow` |
| Scanned resumes need OCR | Wrong dates/numbers, no heading structure | pymupdf4llm OCRs pages with no text layer using a small built-in model (RapidOCR, no torch); validation still flags garbage; Docling stays optional |
| Ollama extraction inconsistent | Wrong metadata | JSON schema, temperature 0, per-entry calls, grounding check, accuracy tests |
| Two-column templates still mis-parsed | Mixed sections | Router + validation + `needs_review`; snapshot tests on two-column fixtures |
| Streamlit reruns reload models | Slow UI | `st.cache_resource`; tested in M9 |
| Synthetic fixtures too clean | Over-optimistic metrics | Include noisy, scanned, designed and gap/overlap fixtures; optional consented real resumes |
| PII leaking into embeddings | Privacy issue | Masking + unit test asserting no email/phone/name in `embed_text` |
