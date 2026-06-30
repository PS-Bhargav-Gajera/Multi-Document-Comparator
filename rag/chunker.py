import tiktoken
from typing import List, Dict, Any, Optional
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TIKTOKEN_ENCODING
from utils.logger import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE_TOKENS
        self.chunk_overlap = CHUNK_OVERLAP_TOKENS
        self.encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for page in pages:
            page_chunks = self._chunk_page(page)
            chunks.extend(page_chunks)
        logger.info("Generated %d chunks from %d pages", len(chunks), len(pages))
        return chunks

    def _chunk_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = page["text"]
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            char_start = len(self.encoding.decode(tokens[:start]))
            char_end = len(self.encoding.decode(tokens[:end]))

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_name": page["document_name"],
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunk_id": f"{page['document_name']}_p{page['page_number']}_c{chunk_id}",
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_count": len(chunk_tokens),
                },
            })
            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks
