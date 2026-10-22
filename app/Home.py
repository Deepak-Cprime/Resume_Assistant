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

role = db.get_role(conn)
st.subheader("Role")
title = st.text_input("Role title", value=role["title"], key="role_title")
jd_text = st.text_area("Job description", value=role["jd_text"], height=260, key="jd_text")
if st.button("Save", key="save_role", type="primary"):
    db.update_role(conn, title=title, jd_text=jd_text)
    st.success("Saved. The job description is used when answering questions; "
               "editing it never re-indexes any resume.")

stats = overview.corpus_stats(conn)
st.subheader("Corpus")
columns = st.columns(5)
columns[0].metric("Resumes", stats.resumes)
columns[1].metric("Candidates", stats.candidates)
columns[2].metric("Chunks", stats.chunks, help=f"{stats.embedded_chunks} are embedded")
columns[3].metric("Waiting to process", stats.pending)
columns[4].metric("Need attention", stats.failed + stats.needs_review)

candidates = overview.candidate_rows(conn)
if candidates:
    st.dataframe(candidates, hide_index=True)
else:
    st.info("No resumes yet. Add some on the **Upload** page.")
