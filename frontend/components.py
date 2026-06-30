import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime


def render_header(title: str, subtitle: Optional[str] = None):
    st.markdown(
        f"""
        <div style="padding: 1.5rem 0 0.5rem 0;">
            <h1 style="margin: 0; font-size: 2rem;">{title}</h1>
            {f'<p style="color: var(--text-color); opacity: 0.7; margin: 0.25rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress(progress: float, text: str):
    st.progress(progress, text=text)


def render_status_badge(status: str, text: str):
    color = {"success": "#00c853", "error": "#ff1744", "warning": "#ff9100", "info": "#2979ff"}.get(status, "#888")
    st.markdown(
        f"""
        <span style="
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            background: {color}20;
            color: {color};
            border: 1px solid {color}40;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )


def render_chunks(chunks: List[Dict[str, Any]]):
    if not chunks:
        st.info("No chunks retrieved")
        return
    st.markdown("### Retrieved Chunks")
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        score = chunk.get("similarity_score", 0)
        with st.expander(
            f"Chunk {i+1} — {meta['document_name']} (p.{meta['page_number']}) "
            f"— Similarity: {score:.3f}",
            expanded=(i == 0),
        ):
            st.markdown(f"**Document:** {meta['document_name']}")
            st.markdown(f"**Page:** {meta['page_number']}")
            st.markdown(f"**Chunk ID:** `{meta['chunk_id']}`")
            st.markdown(f"**Text:**")
            st.text(chunk["text"])


def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    st.markdown("### Sources")
    for src in sources:
        with st.expander(f"📄 {src['document_name']}"):
            st.markdown(f"**Pages:** {', '.join(map(str, src['page_numbers']))}")
            st.markdown(f"**Chunks:**")
            for cid in src["chunk_ids"]:
                st.code(cid)


def render_comparison_result(result: Dict[str, Any]):
    st.markdown("### Comparison Result")
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)

    sources = result.get("sources", [])
    if sources:
        render_sources(sources)

    chunks = result.get("chunks", [])
    if chunks:
        render_chunks(chunks)


def render_upload_status(files: List[str]):
    if not files:
        st.info("No files uploaded yet")
        return
    st.markdown("### Uploaded Files")
    for f in files:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.4rem 0.6rem;
                background: var(--background-color);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                margin: 0.25rem 0;
            ">
                <span>📄</span>
                <span>{f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
