import tiktoken
from langchain_ollama import OllamaEmbeddings
from config import OLLAMA_BASE_URL, EMBED_MODEL, TIKTOKEN_ENCODING
from utils.logger import get_logger

logger = get_logger(__name__)

_embeddings_cache: "OllamaEmbeddings | None" = None
_MAX_EMBED_TOKENS = 500


class EmbeddingGenerator:
    def __init__(self):
        global _embeddings_cache
        if _embeddings_cache is None:
            logger.info(
                "Initializing Ollama embeddings: model=%s url=%s",
                EMBED_MODEL, OLLAMA_BASE_URL,
            )
            _embeddings_cache = OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
        self.client = _embeddings_cache
        self.encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def _truncate(self, text: str) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > _MAX_EMBED_TOKENS:
            logger.warning(
                "Truncating text from %d to %d tokens for embedding",
                len(tokens), _MAX_EMBED_TOKENS,
            )
            tokens = tokens[:_MAX_EMBED_TOKENS]
            return self.encoding.decode(tokens)
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info("Generating embeddings for %d texts", len(texts))
        truncated = [self._truncate(t) for t in texts]
        embeddings = self.client.embed_documents(truncated)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        truncated = self._truncate(text)
        return self.client.embed_query(truncated)
