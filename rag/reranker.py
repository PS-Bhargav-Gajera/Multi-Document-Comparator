from typing import List, Dict, Any
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class MMRReranker:
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param

    def rerank(
        self,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        if not results:
            return results

        query_vec = np.array(query_embedding, dtype=np.float32)
        doc_vecs = np.array(
            [r.get("embedding", np.zeros_like(query_vec)) for r in results],
            dtype=np.float32,
        )

        if doc_vecs.shape[1] == 0:
            return results[:top_k]

        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        doc_vecs_norm = doc_vecs / doc_norms

        sim_to_query = np.dot(doc_vecs_norm, query_vec)
        selected = []
        candidate_indices = list(range(len(results)))

        for _ in range(min(top_k, len(candidate_indices))):
            if not candidate_indices:
                break
            mmr_scores = []
            for idx in candidate_indices:
                sim_q = sim_to_query[idx]
                if selected:
                    sim_selected = max(
                        np.dot(doc_vecs_norm[idx], doc_vecs_norm[s])
                        for s in selected
                    )
                else:
                    sim_selected = 0
                mmr = self.lambda_param * sim_q - (1 - self.lambda_param) * sim_selected
                mmr_scores.append(mmr)

            best_idx = candidate_indices[np.argmax(mmr_scores)]
            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        reranked = [results[i] for i in selected]
        logger.info("MMR reranked %d results to %d", len(results), len(reranked))
        return reranked
