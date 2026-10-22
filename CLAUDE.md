# Resume RAG Assistant

An AI assistant for interviewers. It indexes many resumes submitted for a job role and answers
questions about them ("who has run Kubernetes in production?", "compare these three candidates
against the JD") with citations to the resume text.

- Assignment project, built to be production-ready later.
- Open source first: sentence-transformers / Ollama-served embedding models, Ollama for the LLM.
- Every external dependency (parser, embedder, LLM, vector store, keyword index, reranker, eval
  judge) sits behind an interface and is chosen in config, so free components can be replaced by
  paid ones without code changes.
- Architecture diagram: `docs/rag_resume_assistant_architecture.excalidraw`.
- Style reference for study guides: `RAG Field Guide.html`.

## Phases

1. **Indexing**: upload → parse → normalize/sections → metadata → adaptive chunking → embed →
   vector + BM25 + metadata stores.
2. **Retrieval + generation**: query understanding → hybrid search → rerank → aggregate per
   candidate → context → LLM answer with citations.
3. **Quality**: golden dataset, retrieval/generation/system metrics, embedding model comparison,
   CI eval gate.

## Working rule: learn first, then build

The user wants to understand every phase end to end before any code is written for it.
Do not write application code unless the user explicitly asks for it.

**Before starting any new phase, draft a study guide for that phase first.** The guide must:

1. Be a detailed, multi-chapter HTML artifact in the style of `RAG Field Guide.html`: chapter
   rail, one chapter per page, light/dark themes, flow diagrams, tables, and interactive
   visualizations where they help understanding.
2. For every step of the phase, list **all the main existing methods**, not just the chosen one,
   with how each works, its strengths and weaknesses, and open-source and paid tool examples.
3. State **which method fits this resume use case and why**, and when you would switch to an
   alternative.
4. Give **concrete examples** using resume content, including examples of the methods that were
   not chosen, so the trade-offs are visible.
5. End with a decision record (step → chosen method → alternatives → switch trigger) and the
   phase's exit criteria.
6. Save the source HTML in `docs/guides/` (for example `docs/guides/phase1_indexing_guide.html`)
   and publish it as an artifact.

Only after the user has read and agreed with the guide does work on that phase begin.

## Guide status

- Phase 1 (Indexing): `docs/guides/phase1_indexing_guide.html` — reviewed and approved; development may start
- Phase 2 (Retrieval + generation): not started
- Phase 3 (Quality): not started

## Phase 1 development plan

Full plan, including the testing phase: `docs/phase1_development_plan.md`. This file is the
source of truth; keep the plan document consistent with it.

Scope decisions for the assignment:

- **One role** with an editable job description. The `roles` table keeps a single row (so
  multi-role works later without a schema change). Editing the JD never triggers re-indexing:
  the JD is only used at query time in Phase 2.
- **UI: Streamlit**, calling the core Python package directly. No Flask/FastAPI layer for now.
  All logic lives in `src/` so a FastAPI layer can be added for production without touching it.
- Do not run installs, scripts, or the app unless the user asks. Write code; the user runs it.

### Streamlit rules

- Load embedders, indexes, and DB connections with `st.cache_resource`; never on every rerun.
- Indexing runs from a button with `st.status` / progress bar, processing documents one by one
  and updating the `documents` status table, so a crash leaves a resumable state.
- The UI only calls functions in `src/pipeline` and `src/stores`; no business logic in pages.

### Project layout

```
app/                    Streamlit
  Home.py               role + editable JD, corpus stats
  pages/1_Upload.py     upload PDFs/DOCX, per-document status, retry failed
  pages/2_Inspect.py    per resume: raw → markdown → sections → metadata → chunks
  pages/3_Index.py      build/rebuild per embedding model, manifest, size stats, delete candidate
src/
  config.py             load configs/config.yaml + env vars into typed settings
  providers/            Protocol interfaces + factory
    base.py             DocumentParser, EmbeddingProvider, LLMProvider, VectorStore, KeywordIndex
    parsers/            pymupdf4llm_parser.py (+ layout.py column-aware), docx_parser.py,
                        docling_parser.py (optional, heavy), router.py
    embedders/          st_embedder.py (sentence-transformers), ollama_embedder.py
    llm/                ollama_llm.py
    vectorstores/       faiss_store.py, numpy_store.py (no extra dependency, same behaviour)
    keyword/            bm25_index.py, tokenizer.py (tech-aware)
  ingestion/            intake.py (validate, hash, dedupe), cleaning.py, validation.py (parse quality)
  structure/            sections.py (synonyms + heuristics), entries.py, dates.py, text.py
  metadata/             schema.py (Pydantic), regex_extract.py, llm_extract.py (optional, grounded),
                        skills.py, pii.py, yoe.py (interval union + gaps), profile.py (assembles all)
  chunking/             chunker.py (section + entry), budgets.py (adaptive), enrich.py (prefix, PII),
                        tokens.py (model-free token estimate)
  stores/               db.py (SQLite schema + repo functions), files.py (raw/markdown paths)
  pipeline/             run.py (orchestrates stages, statuses, idempotency), delete.py, manifest.py,
                        parse_stage.py, structure_stage.py, extract_stage.py, chunk_stage.py,
                        index_stage.py (vectors), keyword_stage.py (BM25); one per stage,
                        run.py calls them in order
configs/
  config.yaml           providers, models, budgets, paths
  sections_resume.yaml  canonical section synonyms
  skills_aliases.yaml   skill normalisation (shared by metadata + BM25 tokenizer)
data/                   raw/, markdown/, profiles/, chunks/, indexes/, app.db   (gitignored)
tests/                  unit tests per module + fixtures/
eval/                   golden_questions.yaml (drafted during Phase 1, used in Phase 3)
```

### Milestones (build in this order; each ends with tests passing)

| # | Milestone | Deliverables | Done when |
|---|-----------|--------------|-----------|
| M0 | Scaffolding | layout above, `requirements.txt`, `config.yaml`, `config.py`, provider Protocols + factory, logging | `import src` works; config loads |
| M1 | Storage + intake | SQLite schema (roles, candidates, documents, experiences, skills, certifications, chunks, index_manifest); validate/hash/store raw files; status table | re-uploading a file creates nothing new |
| M2 | Parsing | pymupdf4llm (column-aware for sidebars) + python-docx parsers, optional Docling, router (chars/page, column detection), validation, cleaning; Markdown saved | 10 sample resumes parse with correct reading order |
| M3 | Structure | section detection (synonyms + heuristics), entry splitting, date normalisation | ≥ 90% tokens in known sections on samples |
| M4 | Metadata | regex contact fields, YOE as interval union, gaps, skill aliases (claimed vs used), Ollama JSON-schema extraction per entry with grounding check, PII masking | field accuracy ≥ 90% on hand-labelled samples |
| M5 | Chunking | section + entry chunker, adaptive per-section budgets stored on the role, structural prefix, profile chunk, parent links, deterministic chunk ids | 0 chunks over model limit; every experience chunk has company + dates |
| M6 | Embedding + vectors | sentence-transformers embedder (bge-small), Ollama embedder (nomic, optional), FAISS `IndexIDMap(IndexFlatIP)` per model, persisted + manifest | two model indexes side by side; counts match DB |
| M7 | Keyword index | bm25s with tech-aware tokenizer, persisted | C#, C++, .NET stay distinct tokens |
| M8 | Orchestration | `pipeline.run` with per-stage status, content-hash skip, re-chunk on budget change, delete-everywhere | idempotency + delete tests pass |
| M9 | Streamlit UI | the four pages above | full flow works from the browser |
| M10 | Testing phase + exit | full test suite, accuracy + UI smoke tests, parse report, manual QA, golden questions | exit criteria in the plan document (section 8) met |

### Testing rules

- Write tests with each milestone; a milestone is done only when its tests pass.
- Default test runs need no network, Ollama or model downloads: use `FakeLLM` / `FakeEmbedder`.
  Real-model tests are marked `ollama` / `models`; slow parsers are marked `slow`.
- Tests use `tmp_path`; never touch the real `data/` directory.
- Test data is fictional (synthetic fixtures with ground-truth YAML); never commit real resumes.

### Defaults

- Python 3.11, `sentence-transformers`, `pymupdf4llm`, `python-docx`, `faiss-cpu`, `bm25s`,
  (`docling` is optional: `requirements-docling.txt`, off by default because it needs torch),
  `pydantic`, `pyyaml`, `ollama`, `streamlit`, `pytest`.
- Ollama: `qwen2.5:7b` (or `llama3.1:8b`) for extraction at temperature 0; `nomic-embed-text`.
- Embedding baseline `BAAI/bge-small-en-v1.5`; optional challenger `nomic-embed-text` via Ollama.
  `BAAI/bge-m3` is out of scope (too heavy for the target laptop).

### Open items

- Test corpus: 15–20 resumes (two-column, scanned, DOCX, one non-English). Real resumes need
  consent; otherwise generate synthetic ones.
