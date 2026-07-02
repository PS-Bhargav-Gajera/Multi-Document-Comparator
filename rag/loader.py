"""
Structured PDF extraction using pdfplumber with layout-aware block detection.
Preserves paragraphs, headings, reading order, and block structure.
No text cleaning — extracted text is stored as-is.
Uses character-position-based word spacing inference.
"""
import warnings
warnings.filterwarnings("ignore", message=".*Visual C\\+\\+ Redistributable.*")
import statistics
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber

from utils.logger import get_logger

logger = get_logger(__name__)

HEADING_SIZE_RATIO = 1.15
LINE_Y_TOLERANCE = 4.0
PARAGRAPH_GAP = 5.0
SPACE_GAP_THRESHOLD = 0.35
MIN_LINE_ALPHA = 3
MAX_LINE_DIGIT_RATIO = 0.6
SUPERSCRIPT_SIZE_RATIO = 0.75
SECTION_HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*[\)\.]?\s+[A-Z]")
SUPERSCRIPT_CLEANUP = re.compile(r"(?<=[a-zA-Z,;])\d+$")


class PDFLoader:
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        logger.info("Loading PDF: %s", path.name)
        self._last_path = path
        pages = self._load_pdfplumber(path)
        logger.info("Loaded %d pages from %s", len(pages), path.name)
        return pages

    def load_all(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        all_pages = []
        for fp in file_paths:
            all_pages.extend(self.load(fp))
        return all_pages

    # ── pdfplumber backend ───────────────────────────────────────────────

    def _load_pdfplumber(self, path: Path) -> List[Dict[str, Any]]:
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_chars = page.chars
                if not raw_chars:
                    text = page.extract_text() or ""
                    pages.append(self._make_page(text, [], page_num, path))
                    continue

                chars = self._filter_chars(raw_chars)
                if not chars:
                    pages.append(self._make_page("", [], page_num, path))
                    continue

                body_size = self._estimate_body_font_size(chars)
                lines = self._chars_to_lines(chars)
                clean_lines = [l for l in lines if self._is_meaningful_line(l)]
                paragraphs = self._lines_to_paragraphs(clean_lines)
                para_meta, full_parts = [], []
                for p_data in paragraphs:
                    cleaned = self._clean_paragraph_text(p_data["text"])
                    is_heading = self._is_heading_line(
                        p_data, body_size
                    )
                    para_meta.append({
                        "text": cleaned,
                        "is_heading": is_heading,
                        "heading": cleaned if is_heading else "",
                    })
                    full_parts.append(cleaned)

                pages.append(self._make_page(
                    "\n\n".join(full_parts), para_meta, page_num, path
                ))
        return pages

    def _make_page(
        self, text: str, paragraphs: List[dict], page_num: int, path: Path
    ) -> Dict[str, Any]:
        return {
            "page_number": page_num,
            "text": text,
            "paragraphs": paragraphs,
            "has_images": False,
            "document_name": path.stem,
            "source_file": path.name,
            "is_scanned": len(text.strip()) < 100,
        }

    # ── Char filtering ───────────────────────────────────────────────────

    def _filter_chars(self, chars: List[Dict]) -> List[Dict]:
        body_size = self._estimate_body_font_size(chars)
        filtered = []
        for c in chars:
            size = c.get("size", 0)
            fontname = c.get("fontname", "")
            if size < body_size * 0.5 or size > body_size * 3.0:
                continue
            filtered.append(c)
        return filtered

    def _estimate_body_font_size(self, chars: List[Dict]) -> float:
        sizes = [round(c.get("size", 10), 1)
                 for c in chars if c.get("size")]
        if not sizes:
            return 11.0
        return Counter(sizes).most_common(1)[0][0]

    # ── Char-to-line pipeline ────────────────────────────────────────────

    def _chars_to_lines(self, chars: List[Dict]) -> List[Dict[str, Any]]:
        sorted_chars = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
        lines, current_line = [], [sorted_chars[0]]
        for c in sorted_chars[1:]:
            if abs(c["top"] - current_line[-1]["top"]) <= LINE_Y_TOLERANCE:
                current_line.append(c)
            else:
                lines.append(self._finalize_char_line(current_line))
                current_line = [c]
        if current_line:
            lines.append(self._finalize_char_line(current_line))
        return lines

    def _finalize_char_line(self, chars: List[Dict]) -> Dict[str, Any]:
        widths = [c["x1"] - c["x0"]
                  for c in chars if (c["x1"] - c["x0"]) > 0]
        median_width = statistics.median(widths) if widths else 5.0

        text, prev_x1 = "", chars[0]["x0"]
        for c in chars:
            gap = c["x0"] - prev_x1
            if gap > median_width * SPACE_GAP_THRESHOLD and text:
                text += " "
            text += c["text"]
            prev_x1 = c["x1"]

        text = text.strip()
        alpha_count = sum(1 for ch in text if ch.isalpha())
        max_size = max(c.get("size", 10) for c in chars)

        return {
            "text": text,
            "top": chars[0]["top"],
            "bottom": max(c.get("doctop", c.get("top", 0)) for c in chars),
            "max_size": max_size,
            "alpha_count": alpha_count,
        }

    def _is_meaningful_line(self, line: Dict[str, Any]) -> bool:
        text = line["text"]
        if not text or len(text) < 2:
            return False
        if line["alpha_count"] < MIN_LINE_ALPHA:
            return False
        digits = sum(1 for c in text if c.isdigit())
        if digits > len(text) * MAX_LINE_DIGIT_RATIO:
            return False
        return True

    # ── Paragraph grouping ───────────────────────────────────────────────

    def _lines_to_paragraphs(self, lines: List[Dict[str, Any]]) -> List[Dict]:
        if not lines:
            return []
        paragraphs, current = [], [lines[0]]
        for line in lines[1:]:
            gap = line["top"] - current[-1]["bottom"]
            if gap > PARAGRAPH_GAP:
                paragraphs.append(self._finalize_paragraph(current))
                current = [line]
            else:
                current.append(line)
        if current:
            paragraphs.append(self._finalize_paragraph(current))
        return paragraphs

    def _finalize_paragraph(self, lines: List[Dict]) -> Dict[str, Any]:
        return {
            "text": " ".join(l["text"] for l in lines),
            "max_size": max(l["max_size"] for l in lines),
            "top": lines[0]["top"],
            "bottom": lines[-1]["bottom"],
        }

    def _clean_paragraph_text(self, text: str) -> str:
        """Remove superscript artifacts and trailing line numbers."""
        text = SUPERSCRIPT_CLEANUP.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Heading detection ────────────────────────────────────────────────

    def _is_heading_line(
        self, para: Dict[str, Any], body_size: float
    ) -> bool:
        text = para["text"].strip()
        if not text:
            return False
        if len(text) > 80:
            return False
        if para["max_size"] > body_size * HEADING_SIZE_RATIO:
            return True
        if SECTION_HEADING_PATTERN.match(text):
            return True
        if re.match(
            r"^(Abstract|Introduction|Related Work|Method(s)?|Experiment(s)?|"
            r"Conclusion|References|Appendix|Background|Approach|Discussion|"
            r"Results|Contributions|Preliminaries|Framework)",
            text.strip()
        ):
            return True
        return False
