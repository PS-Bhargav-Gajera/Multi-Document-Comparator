import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from config import SCANNED_PAGE_CHAR_THRESHOLD, UPLOAD_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader:
    def __init__(self):
        self.scanned_threshold = SCANNED_PAGE_CHAR_THRESHOLD

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        logger.info("Loading PDF: %s", path.name)
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                is_scanned = len(text.strip()) < self.scanned_threshold
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "is_scanned": is_scanned,
                    "document_name": path.stem,
                    "source_file": path.name,
                })
        logger.info("Loaded %d pages from %s", len(pages), path.name)
        return pages

    def load_all(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        all_pages = []
        for fp in file_paths:
            all_pages.extend(self.load(fp))
        return all_pages
