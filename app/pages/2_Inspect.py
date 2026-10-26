"""Inspect: follow one resume through every stage, from the raw file to its chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import ui
from src.pipeline import overview

st.set_page_config(page_title="Inspect", page_icon="\U0001f50d", layout="wide")
settings = ui.get_settings()
conn = ui.get_conn(settings)
ui.mode_switch()
developer = ui.is_developer()

st.title("Inspect a resume")
rows = overview.document_rows(conn)
if not rows:
    st.info("Nothing to inspect yet. Add resumes on the **Upload** page.")
    st.stop()


def label(row: dict) -> str:
    # No status in the label: the selection must survive a resume moving through the pipeline.
    parts = [row["file"], row["candidate"], row["doc_id"][:8]]
    return " · ".join(p for p in parts if p)


labels = {label(r): r["doc_id"] for r in rows}
choice = st.selectbox("Resume", list(labels), key="inspect_doc")
view = overview.document_view(conn, settings, labels[choice])
st.caption(f"Status: {ui.status_text(view.row['status'])}")
for note in view.notes:
    st.info(note)

if developer:
    raw_tab, markdown_tab, sections_tab, metadata_tab, chunks_tab = st.tabs(
        ["Raw file", "Markdown", "Sections", "Metadata", "Chunks"]
    )
else:  # an interviewer needs the candidate's facts and the original file, not the internals
    metadata_tab, raw_tab = st.tabs(["Candidate", "Original file"])

with raw_tab:
    st.write(f"**{view.row['filename']}**: {view.row['doc_type'] or 'unknown type'}"
             + (f", parsed with {view.row['parser'] or 'nothing yet'}" if developer else ""))
    if view.raw_path is not None and view.raw_path.exists():
        st.download_button("Download the original file", view.raw_path.read_bytes(),
                           file_name=view.row["filename"], key="download_raw")
    else:
        st.caption("The original file is not stored (it was rejected at upload).")

if developer:
    with markdown_tab:
        if view.markdown:
            st.caption("What the parser produced after cleaning. "
                       "Section detection starts from this.")
            st.code(view.markdown, language="markdown")
            with st.expander("Rendered preview"):
                st.markdown(view.markdown)
        else:
            st.caption("No Markdown yet.")

    with sections_tab:
        if view.sections:
            st.dataframe(
                [{k: s[k] for k in ("label", "heading", "method", "words")} for s in view.sections],
                hide_index=True,
            )
            for s in view.sections:
                title = f"{s['label']}: {s['heading'] or '(no heading)'} ({s['words']} words)"
                with st.expander(title):
                    st.text(s["text"])
        else:
            st.caption("Sections have not been detected yet.")

with metadata_tab:
    if view.profile is None:
        st.caption("Metadata has not been extracted yet.")
    else:
        st.table([{"field": k, "value": v} for k, v in view.facts.items()])
        st.subheader("Experience")
        st.dataframe(view.experience_table, hide_index=True)
        if view.gaps_table:
            st.warning("Career gaps longer than the threshold:")
            st.dataframe(view.gaps_table, hide_index=True)
        st.subheader("Education")
        st.dataframe(view.education_table, hide_index=True)
        st.subheader("Skills")
        st.dataframe(view.skills_table, hide_index=True)
        if view.certifications_table:
            st.subheader("Certifications")
            st.dataframe(view.certifications_table, hide_index=True)
        for warning in view.profile.warnings:
            st.warning(warning)
        if developer:
            with st.expander("Full JSON"):
                st.json(view.profile.model_dump())

if developer:
    with chunks_tab:
        if not view.chunks:
            st.caption("No chunks for this upload.")
        else:
            st.dataframe(
                [{k: c[k] for k in ("key", "level", "section", "entry", "tokens", "embedded",
                                    "parent", "skills")} for c in view.chunks],
                hide_index=True,
            )
            key = st.selectbox("Show a chunk", [c["key"] for c in view.chunks], key="chunk_choice")
            chunk = next(c for c in view.chunks if c["key"] == key)
            shown, embedded = st.columns(2)
            shown.caption("Text shown to the interviewer and cited")
            shown.code(chunk["text"], language="markdown")
            embedded.caption("Text that is embedded and keyword-indexed (prefix, tags, no PII)")
            embedded.code(chunk["embed_text"], language="markdown")

