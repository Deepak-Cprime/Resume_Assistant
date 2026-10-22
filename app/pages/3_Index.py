"""Index: build, rebuild and check the vector and keyword indexes; remove a candidate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import ui
from src.pipeline import delete, overview

st.set_page_config(page_title="Index", page_icon="\U0001f4da", layout="wide")
settings = ui.get_settings()
conn = ui.get_conn(settings)

st.title("Indexes")
ui.show_flash()

st.subheader("Build")
configured = [m.name for m in settings.embedding.models]
models = st.multiselect("Embedding models", configured, default=[settings.embedding.default],
                        key="models")
build_col, rebuild_col = st.columns(2)
build_clicked = build_col.button("Build / update", key="build_btn", type="primary",
                                 disabled=not models)
rebuild_clicked = rebuild_col.button("Rebuild from scratch", key="rebuild_btn",
                                     disabled=not models,
                                     help="Re-embeds every chunk. Only needed after changing the "
                                          "model or its settings.")
if build_clicked or rebuild_clicked:
    # doc_ids=set(): sync the indexes only; resumes still waiting are processed on Upload
    ui.render_result(ui.run_with_progress(
        conn, settings, doc_ids=set(), models=models, rebuild=rebuild_clicked
    ))

view = overview.index_overview(conn, settings)  # after the build so it shows the new state

st.subheader("Status")
in_sync = all(m.report.ok for m in view.models if m.manifest) and view.keyword_report.ok
if view.embeddable_chunks == 0:
    st.info("No chunks yet. Process some resumes first.")
elif in_sync and any(m.manifest for m in view.models):
    st.success("Every index matches the database.")
else:
    st.warning("Some indexes are missing or out of date. Use **Build / update**.")

st.dataframe(
    [
        {
            "embedding model": m.model,
            "vectors": m.manifest.n_vectors if m.manifest else 0,
            "dimension": m.manifest.dim if m.manifest else "",
            "version": m.manifest.index_version if m.manifest else "not built",
            "built": m.manifest.created_at if m.manifest else "",
            "matches database": "yes" if m.report.ok else "no",
            "missing": len(m.report.missing),
            "stale": len(m.report.stale),
            "changed": len(m.report.changed),
        }
        for m in view.models
    ],
    hide_index=True,
)
km = view.keyword_manifest
st.dataframe(
    [{
        "keyword index": "BM25 (" + (km.tokenizer if km else settings.keyword.tokenizer) + ")",
        "documents": km.n_docs if km else 0,
        "built": km.created_at if km else "not built",
        "matches database": "yes" if view.keyword_report.ok else "no",
    }],
    hide_index=True,
)

st.subheader("Chunk sizes")
if view.sizes:
    frame = pd.DataFrame({"chunks": [c for _, c in view.sizes]}, index=[b for b, _ in view.sizes])
    st.bar_chart(frame)
    st.caption("Tokens per embedded chunk (conservative estimate). The budgets below decide when "
               "an entry is split.")
    st.dataframe(view.budget_rows, hide_index=True)
else:
    st.caption("No embedded chunks yet.")

st.subheader("Remove a candidate")
candidates = overview.candidate_rows(conn)
if not candidates:
    st.caption("No candidates.")
else:
    names = {f"{c['name']} (#{c['candidate_id']}, {c['chunks']} chunks)": c["candidate_id"]
             for c in candidates}
    pick = st.selectbox("Candidate", list(names), key="delete_choice")
    sure = st.checkbox("Yes, delete this candidate everywhere", key="delete_confirm")
    if st.button("Delete", key="delete_btn", disabled=not sure):
        removed = delete.delete_candidate(conn, settings, names[pick])
        vectors = sum(removed.vectors.values())
        ui.flash(f"Deleted {removed.documents} upload(s), {removed.chunks} chunk(s) and "
                 f"{vectors} vector(s).")
        for problem in removed.errors:
            st.warning(problem)
        st.rerun()
