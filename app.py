import streamlit as st
from pathlib import Path
from rag.pipeline import RAGPipeline
from frontend.sidebar import render_sidebar
from frontend.upload import render_upload_page, render_uploaded_files_list, render_file_management
from frontend.compare import render_compare_page, render_stats
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Multi-Document Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_MODE_CSS = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, span {
        color: #e0e0e0 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #333 !important;
    }
    .stButton button {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
    }
    .stButton button[kind="primary"] {
        background-color: #4a6cf7 !important;
        color: white !important;
    }
    .stExpander {
        background-color: #1a1a2e !important;
        border: 1px solid #333 !important;
    }
    .stAlert {
        background-color: #1e1e2e !important;
        border: 1px solid #333 !important;
    }
    .stProgress > div > div {
        background-color: #4a6cf7 !important;
    }
    code {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
    }
    .stRadio > div {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e1e2e !important;
    }
</style>
"""


def init_session_state():
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "ingested" not in st.session_state:
        st.session_state["ingested"] = False
    if "stats" not in st.session_state:
        st.session_state["stats"] = None
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = RAGPipeline()


def main():
    init_session_state()

    if st.session_state.get("dark_mode", False):
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

    page = render_sidebar()

    pipeline = st.session_state["pipeline"]

    uploaded_files = render_uploaded_files_list()
    st.session_state["uploaded_files"] = uploaded_files

    if page == "Upload Documents":
        new_files = render_upload_page()
        if new_files:
            st.session_state["uploaded_files"] = render_uploaded_files_list()
        render_file_management(uploaded_files)

    elif page == "Compare Documents":
        render_compare_page(pipeline)
        render_stats()

    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    with col2:
        dark_mode = st.checkbox(
            "🌙 Dark Mode",
            value=st.session_state.get("dark_mode", False),
            key="dark_mode_toggle",
        )
        if dark_mode != st.session_state.get("dark_mode", False):
            st.session_state["dark_mode"] = dark_mode
            st.rerun()


if __name__ == "__main__":
    main()
