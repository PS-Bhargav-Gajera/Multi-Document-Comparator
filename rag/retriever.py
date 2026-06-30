from typing import List, Dict, Any
import tiktoken
from rag.embeddings import EmbeddingGenerator
from rag.vectordb import VectorDatabase
from rag.reranker import MMRReranker
from config import FETCH_K, TOP_K, MAX_CONTEXT_TOKENS, TIKTOKEN_ENCODING
from utils.logger import get_logger

logger = get_logger(__name__)


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

        reranked = self.reranker.rerank(
            query_embedding=query_embedding,
            results=initial_results,
            top_k=TOP_K,
        )

        compressed = self._compress_context(reranked)
        logger.info("Retrieved %d chunks after compression", len(compressed))
        return compressed

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
