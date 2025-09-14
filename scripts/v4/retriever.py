import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import normalize

def get_context(query: str,
                embed_model,
                doc_embeddings: np.ndarray,
                documents: List[str],
                doc_ids: List[str],
                top_k: int = 6,
                threshold: float = 0.2) -> Tuple[Optional[str], Optional[List[str]]]:
    """Retourne le contexte (concaténation) et la liste des sources."""
    qv = embed_model.encode([query], convert_to_numpy=True)
    qv = normalize(qv)
    sims = np.dot(doc_embeddings, qv.T).squeeze()

    idx_ok = np.where(sims >= threshold)[0]
    if len(idx_ok) == 0:
        return None, None

    top_idx = idx_ok[np.argsort(sims[idx_ok])[::-1][:top_k]]
    contexts = [documents[i] for i in top_idx]
    sources = [doc_ids[i] for i in top_idx]
    return "\n\n".join(contexts), sources