"""Upload: add PDF/DOCX resumes, process them, and deal with the ones that need attention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import ui
from src.pipeline import delete, overview, uploads
from src.pipeline import run as pipeline

st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
settings = ui.get_settings()
conn = ui.get_conn(settings)
ui.mode_switch()
developer = ui.is_developer()

st.title("Upload resumes")
ui.show_flash()
files = st.file_uploader(
    f"PDF or DOCX resumes (up to {settings.ingestion.max_file_mb} MB and "
    f"{settings.ingestion.max_pages} pages each)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="uploader",
)

stats = overview.corpus_stats(conn)
needs_attention = stats.failed + stats.needs_review
col_upload, col_process, col_retry = st.columns(3)
upload_clicked = col_upload.button(
    "Upload and process", key="upload_btn", type="primary", disabled=not files
)
process_clicked = col_process.button("Process pending resumes", key="process_btn")
retry_clicked = col_retry.button(
    "Retry failed and needs-review", key="retry_btn", disabled=needs_attention == 0
)

if upload_clicked and files:
    results = uploads.ingest_files(conn, settings, [(f.name, f.getvalue()) for f in files])
    st.dataframe(
        [{"file": r.filename, "result": uploads.outcome(r)} for r in results], hide_index=True
    )
    process_clicked = True  # an upload is followed by processing straight away
if retry_clicked:
    pipeline.retry_failed(conn)
    process_clicked = True
if process_clicked:
    ui.render_result(ui.run_with_progress(conn, settings))

st.subheader("Documents")
rows = overview.document_rows(conn)
if not rows:
    st.info("Nothing uploaded yet.")
else:
    statuses = sorted({r["status"] for r in rows})
    # No fixed key: the options change as documents move through the pipeline, and a keyed
    # widget would keep a stale selection and hide the rows that just changed status.
    shown = st.multiselect("Show statuses", statuses, default=statuses)
    st.dataframe(
        [{**r, "status": ui.status_text(r["status"])} for r in rows if r["status"] in shown],
        column_order=(
            ["file", "status", "candidate", "type", "parser", "note", "uploaded"] if developer
            else ["file", "status", "candidate", "note"]
        ),
        hide_index=True,
    )

flagged = [r for r in rows if r["status"] in ("failed", "needs_review")]
if flagged:
    st.subheader("Needs attention")
    for r in flagged:
        st.warning(f"**{r['file']}** ({r['status']}): {r['note']}")

if rows:
    with st.expander("Remove an upload"):
        labels = {f"{r['file']} [{r['status']}] {r['doc_id'][:8]}": r["doc_id"] for r in rows}
        choice = st.selectbox("Upload", list(labels), key="remove_choice")
        confirmed = st.checkbox(
            "Yes, remove it (and its candidate, if it has one)", key="remove_confirm"
        )
        if st.button("Remove", key="remove_btn", disabled=not confirmed):
            removed = delete.delete_document(conn, settings, labels[choice])
            ui.flash(f"Removed {removed.documents} upload(s) and {removed.chunks} chunk(s).")
            st.rerun()
