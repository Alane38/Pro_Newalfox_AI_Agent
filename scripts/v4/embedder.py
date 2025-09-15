from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Tuple

EMB_FILE = Path("../../data/embeddings.npy")

def load_or_compute_embeddings(documents: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, SentenceTransformer]:
    """Charge des embeddings si présents sinon les calcule, normalise et retourne (embeddings, model)."""
    model = SentenceTransformer(model_name)
    if EMB_FILE.exists():
        try:
            emb = np.load(EMB_FILE)
            emb = normalize(emb)
            return emb, model
        except Exception:
            pass
    if not documents:
        raise ValueError("The documents list is empty. Please provide a non-empty list of documents.")
    emb = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    emb = normalize(emb)
    np.save(EMB_FILE, emb)
    return emb, model