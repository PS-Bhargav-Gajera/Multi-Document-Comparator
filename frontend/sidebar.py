import streamlit as st
from pathlib import Path
from typing import List
from config import UPLOAD_DIR, CHROMA_DB_DIR


def render_sidebar() -> str:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/combo-chart.png",
            width=64,
        )
        st.markdown("## Document Comparator")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=["Upload Documents", "Compare Documents"],
            index=0,
            key="navigation",
        )

        st.markdown("---")
        st.markdown("### Storage Info")

        upload_path = Path(UPLOAD_DIR)
        uploaded_files = list(upload_path.glob("*.pdf")) if upload_path.exists() else []
        st.markdown(f"- **Uploaded PDFs:** {len(uploaded_files)}")

        chroma_path = Path(CHROMA_DB_DIR)
        if chroma_path.exists():
            db_files = list(chroma_path.glob("*"))
            st.markdown(f"- **ChromaDB size:** {len(db_files)} files")

        st.markdown("---")
        st.markdown("### Settings")

        st.markdown("**Similarity Metric:** Cosine")
        st.markdown("**Top-K:** 8")
        st.markdown("**Embedding Model:** mxbai-embed-large")
        st.markdown("**LLM:** gpt-oss-120b (OpenRouter)")

        st.markdown("---")
        st.markdown(
            "<small style='opacity:0.6'>Multi-Document Comparator v1.0</small>",
            unsafe_allow_html=True,
        )

    return page
