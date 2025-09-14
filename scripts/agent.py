from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from llama_cpp import Llama

# --- 1. Charger documents ---
DATA_DIR = Path("../data")
documents = [f.read_text(encoding="utf-8") for f in DATA_DIR.glob("*.txt")]
doc_ids = [f.name for f in DATA_DIR.glob("*.txt")]

# --- 2. Créer les embeddings des documents ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # rapide et efficace
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

# --- 3. Charger le modèle LLaMA local ---
MODEL_PATH = "../models/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=4)

# --- 4. Fonction pour récupérer les meilleurs passages ---
def get_context(query, top_k=6, threshold=0.2): # top_k = nombre de passages, threshold = seuil de similarité
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    similarities = np.dot(doc_embeddings, query_vec.T).squeeze()
    
    # Filtrer par seuil
    best_idx = np.where(similarities >= threshold)[0]
    if len(best_idx) == 0:
        return None, None  # Aucun document pertinent
    
    # Récupérer les top-k
    top_idx = best_idx[np.argsort(similarities[best_idx])[::-1][:top_k]]
    contexts = [documents[i] for i in top_idx]
    sources = [doc_ids[i] for i in top_idx]
    return "\n\n".join(contexts), sources

# --- 5. Fonction pour poser une question ---
def ask_question(query, max_tokens=512):
    context, sources = get_context(query)
    if context is None:
        return "Je n'ai pas d'information fiable pour répondre à cette question.", None

    prompt = (
        f"Répond professionnellement uniquement avec les informations présentes dans le contexte ci-dessous.\n\n"
        f"Contexte:\n{context}\n\n"
        f"Question: {query}\nRéponse:"
    )
    output = llm(prompt, max_tokens=max_tokens)
    answer = output["choices"][0]["text"].strip()
    return answer, sources

# --- 6. Exemple d'utilisation ---
if __name__ == "__main__":
    print("Tape 'exit' ou 'quit' pour quitter.")
    while True:
        q = input("Question: ")
        if q.lower() in ("exit", "quit"):
            break
        answer, sources = ask_question(q)
        print("Réponse de l'IA :", answer)
        # if sources:
        #     print("Source(s) utilisée(s) :", ", ".join(sources))
