"""Home: the role and its editable job description, and the state of the corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import ui
from src.pipeline import overview
from src.stores import db

st.set_page_config(page_title="Resume RAG Assistant", page_icon="\U0001f4c4", layout="wide")
settings = ui.get_settings()
conn = ui.get_conn(settings)

st.title("Resume RAG Assistant")
st.caption("Index the resumes submitted for one role, then ask questions about them.")

ui.mode_switch()
developer = ui.is_developer()

role = db.get_role(conn)
has_jd = bool(role["jd_text"].strip())
st.subheader("Role (optional)")
if has_jd:
    st.caption(f"**{role['title'] or 'Untitled role'}**: a job description is set and will be "
               "used when answering questions.")
else:
    st.caption("No job description yet. You can index and search resumes without one; add it "
               "to compare candidates against a role.")
with st.expander("Role and job description", expanded=False):
    title = st.text_input("Role title (optional)", value=role["title"], key="role_title")
    jd_text = st.text_area("Job description (optional)", value=role["jd_text"], height=260,
                           key="jd_text")
    if st.button("Save", key="save_role", type="primary"):
        db.update_role(conn, title=title.strip(), jd_text=jd_text.strip())
        st.success("Saved. The job description is used when answering questions; "
                   "editing it never re-indexes any resume.")

stats = overview.corpus_stats(conn)
st.subheader("Corpus")
if developer:
    columns = st.columns(5)
    columns[0].metric("Resumes", stats.resumes)
    columns[1].metric("Candidates", stats.candidates)
    columns[2].metric("Chunks", stats.chunks, help=f"{stats.embedded_chunks} are embedded")
    columns[3].metric("Waiting to process", stats.pending)
    columns[4].metric("Need attention", stats.failed + stats.needs_review)
else:
    columns = st.columns(3)
    columns[0].metric("Resumes", stats.resumes)
    columns[1].metric("Candidates", stats.candidates)
    columns[2].metric("Need attention", stats.failed + stats.needs_review)

candidates = overview.candidate_rows(conn)
if candidates:
    shown = candidates if developer else [
        {k: c[k] for k in ("name", "headline", "years", "seniority")} for c in candidates
    ]
    st.dataframe(shown, hide_index=True)
else:
    st.info("No resumes yet. Add some on the **Upload** page.")
