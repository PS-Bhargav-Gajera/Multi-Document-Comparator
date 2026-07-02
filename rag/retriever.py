from typing import List, Dict, Any, Optional
import re
import tiktoken
from rag.embeddings import EmbeddingGenerator
from rag.vectordb import VectorDatabase
from rag.reranker import MMRReranker
from config import FETCH_K, TOP_K, MAX_CONTEXT_TOKENS, TIKTOKEN_ENCODING
from utils.logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.35
MIN_CHUNKS_PER_DOC = 2

FIG_REF_PATTERNS = re.compile(
    r"(Figure\s+\d+|Table\s+\d+|Algorithm\s+\d+|Eq\.\s*\(?\d+|Equation\s+\d+)",
    re.IGNORECASE,
)
REFERENCE_PATTERNS = re.compile(
    r"(References|Bibliography|Acknowledgments|Appendix)",
    re.IGNORECASE,
)
GARBAGE_PATTERNS = re.compile(
    r"(^[^a-zA-Z]*$|^\d+[\.\)]\s*$|^\s*[a-zA-Z]\s*$)",
)


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_db: VectorDatabase,
        reranker: MMRReranker,
    ):
        self.embedder = embedder
        self.vector_db = vector_db
        self.reranker = reranker
        self.encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)

        initial_results = self.vector_db.similarity_search(
            query_embedding, k=FETCH_K
        )

        if not initial_results:
            logger.warning("No results found for query: %s", query[:50])
            return []

        filtered = self._filter_low_quality(initial_results)

        balanced = self._balance_by_document(filtered)

        reranked = self.reranker.rerank(
            query_embedding=query_embedding,
            results=balanced,
            top_k=TOP_K,
        )

        reranked = self._ensure_min_per_doc(reranked, filtered)

        compressed = self._compress_context(reranked)
        logger.info("Retrieved %d chunks after compression", len(compressed))
        return compressed

    def _filter_low_quality(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        filtered = []
        for r in results:
            text = r["text"]
            if GARBAGE_PATTERNS.match(text):
                continue
            words = text.split()
            if len(words) < 30:
                continue
            alpha = sum(1 for c in text if c.isalpha())
            if alpha / max(len(text), 1) < 0.3:
                continue
            filtered.append(r)
        return filtered

    def _balance_by_document(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        doc_groups: dict[str, list] = {}
        for r in results:
            doc = r["metadata"]["document_name"]
            doc_groups.setdefault(doc, []).append(r)

        num_docs = len(doc_groups)
        if num_docs < 2:
            return results

        per_doc = max(FETCH_K // num_docs, MIN_CHUNKS_PER_DOC)
        balanced: list = []
        for doc, doc_results in doc_groups.items():
            balanced.extend(doc_results[:per_doc])
        return balanced

    def _ensure_min_per_doc(
        self, reranked: List[Dict[str, Any]], all_initial: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        doc_counts: dict[str, int] = {}
        for r in reranked:
            doc = r["metadata"]["document_name"]
            doc_counts[doc] = doc_counts.get(doc, 0) + 1

        num_docs = len(doc_counts)
        if num_docs < 2:
            return reranked

        underrep = [d for d, c in doc_counts.items() if c < MIN_CHUNKS_PER_DOC]
        if not underrep:
            return reranked

        reranked_cids = {r["metadata"]["chunk_id"] for r in reranked}
        pool: list = list(reranked)

        for doc in underrep:
            need = MIN_CHUNKS_PER_DOC - doc_counts[doc]
            extras = [
                r for r in all_initial
                if r["metadata"]["document_name"] == doc
                and r["metadata"]["chunk_id"] not in reranked_cids
            ]
            pool.extend(extras[:need])

        pool.sort(key=lambda r: r.get("similarity_score", 0), reverse=True)
        result = pool[:TOP_K]
        return result

    def _compress_context(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        compressed = []
        total_tokens = 0
        for r in results:
            tokens = len(self.encoding.encode(r["text"]))
            if total_tokens + tokens > MAX_CONTEXT_TOKENS:
                break
            compressed.append(r)
            total_tokens += tokens
        return compressed
