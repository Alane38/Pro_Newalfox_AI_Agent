from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import numpy as np

# --- 1. Charger documents ---
DATA_DIR = Path("../data")
documents = [f.read_text(encoding="utf-8") for f in DATA_DIR.glob("*.txt")]

# --- 2. Créer le TF-IDF pour recherche de contexte ---
vectorizer = TfidfVectorizer(max_features=5000)  # limite mémoire
tfidf_matrix = vectorizer.fit_transform(documents)

# --- 3. Charger le modèle LLaMA local ---
MODEL_PATH = "../models/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=6)  # utilise plus de threads CPU

# --- 4. Fonction pour récupérer les top-k passages ---
def get_context(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return "\n\n---\n\n".join([documents[i] for i in top_indices])

# --- 5. Fonction pour poser une question ---
def ask_question(query):
    context = get_context(query, top_k=3)
    prompt = (
        f"Tu es un assistant expert. Utilise le contexte ci-dessous pour répondre "
        f"à la question de manière claire et concise.\n\n"
        f"Contexte:\n{context}\n\n"
        f"Question: {query}\nRéponse:"
    )
    output = llm(prompt, max_tokens=256)  # réduit le temps de réponse
    return output["choices"][0]["text"].strip()

# --- 6. Boucle interactive ---
if __name__ == "__main__":
    print("Tape 'exit' ou 'quit' pour quitter.")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        answer = ask_question(q)
        print("Réponse de l'IA:", answer)
