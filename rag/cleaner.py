import re
import unicodedata
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    def clean(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Cleaning %d pages", len(pages))
        cleaned = []
        seen_texts = set()
        for page in pages:
            text = page["text"]
            text = self._normalize_unicode(text)
            text = self._remove_invisible_chars(text)
            text = self._normalize_line_breaks(text)
            text = self._remove_duplicate_whitespace(text)
            text = self._fix_broken_hyphenation(text)
            text = self._remove_headers_footers(text)
            if not text.strip():
                continue
            text_normalized = re.sub(r"\s+", " ", text).strip()
            if text_normalized in seen_texts:
                continue
            seen_texts.add(text_normalized)
            page["text"] = text.strip()
            cleaned.append(page)
        logger.info("After cleaning: %d pages", len(cleaned))
        return cleaned

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _remove_invisible_chars(self, text: str) -> str:
        return re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u2028-\u202f\ufeff]", "", text)

    def _normalize_line_breaks(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\n{3,}", "\n\n", text)

    def _remove_duplicate_whitespace(self, text: str) -> str:
        return re.sub(r"[ \t]+", " ", text)

    def _fix_broken_hyphenation(self, text: str) -> str:
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def _remove_headers_footers(self, text: str) -> str:
        lines = text.split("\n")
        if len(lines) < 3:
            return text
        candidate_header = lines[0].strip()
        candidate_footer = lines[-1].strip()
        if candidate_header.isdigit() or len(candidate_header) < 60:
            lines = lines[1:]
        if len(lines) > 0 and (candidate_footer.isdigit() or len(candidate_footer) < 60):
            lines = lines[:-1]
        return "\n".join(lines)
