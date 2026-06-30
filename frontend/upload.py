import streamlit as st
from pathlib import Path
from typing import List, Optional
from config import UPLOAD_DIR, MAX_FILE_SIZE_MB, MAX_FILES
from frontend.components import render_header, render_progress, render_status_badge
from utils.logger import get_logger

logger = get_logger(__name__)


def render_upload_page() -> Optional[List[str]]:
    render_header(
        "Upload Documents",
        f"Upload up to {MAX_FILES} PDF files (max {MAX_FILE_SIZE_MB} MB each)",
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Select PDF documents to analyze. Maximum {MAX_FILES} files.",
    )

    if not uploaded_files:
        st.info("👆 Please upload PDF documents to begin.")
        return None

    file_paths = []

    for f in uploaded_files:
        save_path = UPLOAD_DIR / f.name

        if save_path.exists():
            st.warning(f"`{f.name}` already exists — skipping duplicate.")
            file_paths.append(str(save_path))
            continue

        with open(save_path, "wb") as buf:
            buf.write(f.getbuffer())

        file_paths.append(str(save_path))
        st.success(f"✅ `{f.name}` saved successfully.")

    return file_paths


def render_uploaded_files_list() -> List[str]:
    upload_dir = Path(UPLOAD_DIR)
    files = sorted(upload_dir.glob("*.pdf")) if upload_dir.exists() else []
    return [str(f) for f in files]


def render_file_management(existing_files: List[str]):
    if not existing_files:
        return

    st.markdown("### Manage Uploaded Files")
    for fp in existing_files:
        name = Path(fp).name
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"📄 `{name}`")
        if col2.button("🗑️", key=f"del_{name}"):
            Path(fp).unlink(missing_ok=True)
            st.rerun()
