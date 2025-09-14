from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# --- 1. Charger documents ---
DATA_DIR = Path("../data")
documents = [f.read_text(encoding="utf-8") for f in DATA_DIR.glob("*.txt")]

# --- 2. Créer le TF-IDF pour recherche de contexte ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# --- 3. Charger le modèle LLaMA local ---
MODEL_PATH = "../models/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=4)

# --- 4. Fonction pour récupérer le meilleur passage ---
def get_context(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    best_idx = similarities.argmax()
    return documents[best_idx]

# --- 5. Fonction pour poser une question ---
def ask_question(query):
    context = get_context(query)
    prompt = f"Utilise ce contexte pour répondre à la question :\n{context}\n\nQuestion: {query}\nRéponse:"
    output = llm(prompt, max_tokens=512)
    return output["choices"][0]["text"]

# --- 6. Exemple d'utilisation ---
if __name__ == "__main__":
    while True:
        q = input("Question: ")
        if q.lower() in ("exit", "quit"):
            break
        answer = ask_question(q)
        print("Réponse de l'IA :", answer)
