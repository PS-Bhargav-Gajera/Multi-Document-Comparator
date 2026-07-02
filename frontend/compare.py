import streamlit as st
from typing import List, Optional, Dict, Any
from frontend.components import (
    render_header,
    render_comparison_result,
    render_upload_status,
    render_progress,
)
from rag.pipeline import RAGPipeline
from security.prompt_guard import PromptGuard
from security.output_filter import OutputFilter
from utils.logger import get_logger

logger = get_logger(__name__)


def render_compare_page(pipeline: RAGPipeline):
    render_header(
        "Compare Documents",
        "Ask questions about your uploaded documents",
    )

    guard = PromptGuard()
    output_filter = OutputFilter()

    uploaded_files = st.session_state.get("uploaded_files", [])

    if not uploaded_files:
        st.warning("No documents uploaded. Go to **Upload Documents** first.")
        return

    render_upload_status(uploaded_files)

    st.markdown("---")

    if st.button("🔄 Ingest All Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents..."):
            render_progress(0.3, "Loading PDFs...")
            result = pipeline.ingest(uploaded_files)
            st.session_state["ingested"] = True
            st.session_state["stats"] = result
            st.success(
                f"✅ Ingested {result['chunks']} chunks "
                f"from {result['pages']} pages across {len(uploaded_files)} documents."
            )

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        pipeline.clear_all()
        st.session_state["ingested"] = False
        st.session_state["stats"] = None
        st.success("Vector database cleared.")
        st.rerun()

    st.markdown("---")

    query = st.text_input(
        "💬 Ask a question about your documents",
        placeholder="e.g., What are the main similarities and differences between these documents?",
        disabled=not st.session_state.get("ingested", False),
    )

    if query:
        if not guard.validate(query):
            st.error("Your question was flagged as potentially unsafe. Please rephrase.")
            return

        with st.spinner("Analyzing documents..."):
            result = pipeline.query(query)
            result["answer"] = output_filter.filter(result["answer"])

        if result and result.get("chunks"):
            render_comparison_result(result)
        else:
            st.warning("No relevant information found in the uploaded documents.")


def render_stats():
    stats = st.session_state.get("stats")
    if stats:
        with st.expander("📊 Ingestion Statistics", expanded=False):
            st.markdown(f"- **Pages loaded:** {stats['pages']}")
            st.markdown(f"- **Chunks generated:** {stats['chunks']}")
            st.markdown(f"- **Total chunks in DB:** {stats['total_chunks']}")
