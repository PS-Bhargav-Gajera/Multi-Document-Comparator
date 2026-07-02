"""
Semantic chunker — sentence-aware, heading-preserving, with quality filtering.

Strategy:
1. Pages provide structured paragraphs with heading detection
2. Split paragraphs into sentences using regex
3. Group sentences until target token range reached
4. Never split mid-sentence
5. Preserve heading metadata
6. Overlap via sentence-level sliding window

Target: 400-700 tokens per chunk, 50-100 token overlap
"""
import re
import tiktoken
from typing import List, Dict, Any, Optional
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TIKTOKEN_ENCODING
from utils.logger import get_logger

logger = get_logger(__name__)

MIN_CHUNK_TOKENS = 250
TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 700
SENTENCE_SPLITTER = re.compile(
    r"(?<!\b\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\;)\s+"
)
NEWLINE_SPLITTER = re.compile(r"\n\s*\n")


class SemanticChunker:
    def __init__(self):
        self.target_min = max(CHUNK_SIZE_TOKENS, TARGET_MIN_TOKENS)
        self.target_max = TARGET_MAX_TOKENS
        self.overlap = max(CHUNK_OVERLAP_TOKENS, 50)
        self.encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_id_counter = 0

        for page in pages:
            page_chunks, chunk_id_counter = self._chunk_page(
                page, chunk_id_counter
            )
            chunks.extend(page_chunks)

        logger.info("Generated %d chunks from %d pages", len(chunks), len(pages))
        return chunks

    def _chunk_page(
        self, page: Dict[str, Any], start_id: int
    ) -> tuple[List[Dict[str, Any]], int]:
        text = page["text"]
        paragraphs = page.get("paragraphs", [])
        doc_name = page["document_name"]
        page_num = page["page_number"]
        source_file = page.get("source_file", doc_name)

        if not paragraphs:
            paragraphs = self._fallback_paragraphs(text)

        sentences = self._paragraphs_to_sentences(paragraphs)
        if not sentences:
            return [], start_id

        return self._chunk_sentences(
            sentences, doc_name, page_num, source_file, start_id
        )

    def _paragraphs_to_sentences(
        self, paragraphs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        sentences = []
        current_heading = ""
        for para in paragraphs:
            if para.get("is_heading"):
                current_heading = para["text"]
                sentences.append({
                    "text": para["text"],
                    "is_heading": True,
                    "heading": para["text"],
                })
                continue

            para_sentences = SENTENCE_SPLITTER.split(para["text"])
            for sent in para_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sentences.append({
                    "text": sent,
                    "is_heading": False,
                    "heading": current_heading,
                })
        return sentences

    def _chunk_sentences(
        self,
        sentences: List[Dict[str, Any]],
        doc_name: str,
        page_num: int,
        source_file: str,
        start_id: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        chunks = []
        chunk_id = start_id
        i = 0

        while i < len(sentences):
            if sentences[i].get("is_heading") and i + 1 < len(sentences):
                heading_sent = sentences[i]
                content_start = i + 1
            else:
                heading_sent = None
                content_start = i

            selected = []
            selected_tokens = 0

            if heading_sent:
                ht = len(self.encoding.encode(heading_sent["text"]))
                selected.append(heading_sent)
                selected_tokens += ht

            for j in range(content_start, len(sentences)):
                s = sentences[j]
                st = len(self.encoding.encode(s["text"]))

                if s.get("is_heading"):
                    break

                if selected_tokens + st > self.target_max and selected_tokens >= self.target_min:
                    break

                selected.append(s)
                selected_tokens += st

            if selected:
                chunk_text = " ".join(s["text"] for s in selected)
                chunk_heading = ""
                for s in selected:
                    if s.get("is_heading") and s["text"].strip():
                        chunk_heading = s["text"]
                        break
                    if s["heading"]:
                        chunk_heading = s["heading"]

                chunk = self._make_chunk(
                    chunk_text, doc_name, page_num, source_file,
                    chunk_heading, chunk_id, len(selected)
                )
                if self._passes_quality(chunk):
                    chunks.append(chunk)
                chunk_id += 1

            overlap_count = self._compute_sentence_overlap(selected, sentences, i)
            i += max(1, len(selected) - overlap_count)
            if i >= len(sentences):
                break

        return chunks, chunk_id

    def _compute_sentence_overlap(
        self, selected: List[Dict], all_sentences: List[Dict], start_idx: int
    ) -> int:
        overlap_tokens = 0
        count = 0
        for s in reversed(selected):
            if s.get("is_heading"):
                continue
            st = len(self.encoding.encode(s["text"]))
            if overlap_tokens + st > self.overlap:
                if count == 0:
                    count = 1
                break
            overlap_tokens += st
            count += 1
        return count

    def _make_chunk(
        self,
        text: str,
        doc_name: str,
        page_num: int,
        source_file: str,
        heading: str,
        chunk_id: int,
        num_sentences: int,
    ) -> Dict[str, Any]:
        token_count = len(self.encoding.encode(text))
        return {
            "text": text,
            "metadata": {
                "document_name": doc_name,
                "source_file": source_file,
                "page_number": page_num,
                "chunk_id": f"{doc_name}_p{page_num}_c{chunk_id}",
                "heading": heading,
                "token_count": token_count,
                "num_sentences": num_sentences,
            },
        }

    def _passes_quality(self, chunk: Dict[str, Any]) -> bool:
        text = chunk["text"].strip()
        if not text:
            return False
        tokens = chunk["metadata"]["token_count"]
        if tokens < MIN_CHUNK_TOKENS:
            return False
        words = text.split()
        if len(words) < 40:
            return False
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / max(len(text), 1) < 0.3:
            return False
        uc_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if uc_words > len(words) * 0.7:
            return False

        first_word = words[0].lower() if words else ""
        if first_word in ("figure", "table", "algorithm", "fig.", "algorithms"):
            return False

        citation_density = sum(1 for w in words if re.match(r"^\[\d+\]$", w))
        if citation_density > len(words) * 0.15:
            return False

        if re.match(r"^\[", text):
            return False

        space_ratio = text.count(" ") / max(len(text), 1)
        if space_ratio < 0.05:
            return False

        ascii_ratio = sum(1 for c in text if ' ' <= c <= '~') / max(len(text), 1)
        if ascii_ratio < 0.7:
            return False

        return True

    def _fallback_paragraphs(
        self, text: str
    ) -> List[Dict[str, Any]]:
        parts = NEWLINE_SPLITTER.split(text)
        return [
            {"text": p.strip(), "is_heading": False, "heading": ""}
            for p in parts
            if p.strip()
        ]
