import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    def __init__(self):
        self.db_dir = Path(CHROMA_DB_DIR)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _store_path(self, name: str) -> Path:
        return self.db_dir / f"{CHROMA_COLLECTION_NAME}_{name}"

    def _load(self):
        emb_path = self._store_path("embeddings.npy")
        meta_path = self._store_path("metadata.json")
        texts_path = self._store_path("texts.json")
        ids_path = self._store_path("ids.json")

        if emb_path.exists() and meta_path.exists() and texts_path.exists() and ids_path.exists():
            self.embeddings = np.load(str(emb_path)).astype(np.float32)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata: List[Dict[str, Any]] = json.load(f)
            with open(texts_path, "r", encoding="utf-8") as f:
                self.texts: List[str] = json.load(f)
            with open(ids_path, "r", encoding="utf-8") as f:
                self.ids: List[str] = json.load(f)
            logger.info("Loaded vector store with %d entries", len(self.ids))
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.metadata = []
            self.texts = []
            self.ids = []
            logger.info("Initialized empty vector store")

    def _save(self):
        self.db_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._store_path("embeddings.npy")), self.embeddings)
        with open(self._store_path("metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, default=str)
        with open(self._store_path("texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)
        with open(self._store_path("ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False)

    def add_chunks(
        self, chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        existing_ids = set(self.ids)
        new_texts: list[str] = []
        new_metadatas: list[dict] = []
        new_ids: list[str] = []
        new_embeddings: list[list[float]] = []

        for i, c in enumerate(chunks):
            chunk_id = c["metadata"]["chunk_id"]
            if chunk_id in existing_ids:
                logger.warning("Skipping duplicate chunk: %s", chunk_id)
                continue
            existing_ids.add(chunk_id)
            new_texts.append(c["text"])
            new_metadatas.append(c["metadata"])
            new_ids.append(chunk_id)
            if embeddings is not None:
                new_embeddings.append(embeddings[i])

        if not new_ids:
            logger.info("No new chunks to add (all duplicates)")
            return

        emb_array = np.array(new_embeddings, dtype=np.float32)

        if self.embeddings.size == 0:
            self.embeddings = emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, emb_array])

        self.texts.extend(new_texts)
        self.metadata.extend(new_metadatas)
        self.ids.extend(new_ids)
        self._save()
        logger.info("Added %d new chunks (%d duplicates skipped)", len(new_ids), len(chunks) - len(new_ids))

    def similarity_search(
        self, query_embedding: List[float], k: int = 20
    ) -> List[Dict[str, Any]]:
        if self.embeddings.size == 0 or len(self.ids) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        doc_vecs = self.embeddings

        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        doc_vecs_norm = doc_vecs / doc_norms

        similarities = np.dot(doc_vecs_norm, query_vec)
        top_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "embedding": self.embeddings[idx].tolist(),
                "similarity_score": float(similarities[idx]),
            })

        return results

    def delete_collection(self):
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.metadata = []
        self.texts = []
        self.ids = []
        for f in self.db_dir.glob(f"{CHROMA_COLLECTION_NAME}_*"):
            f.unlink(missing_ok=True)
        logger.info("Deleted vector store collection")

    def get_collection_stats(self) -> dict:
        return {"total_chunks": len(self.ids)}

    def delete_document(self, document_name: str):
        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("document_name") != document_name
        ]
        removed = len(self.ids) - len(keep_indices)
        if removed == 0:
            return

        self.embeddings = self.embeddings[keep_indices]
        self.texts = [self.texts[i] for i in keep_indices]
        self.metadata = [self.metadata[i] for i in keep_indices]
        self.ids = [self.ids[i] for i in keep_indices]
        self._save()
        logger.info("Deleted %d chunks for document: %s", removed, document_name)
