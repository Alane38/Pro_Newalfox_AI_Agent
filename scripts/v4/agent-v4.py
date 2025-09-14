import argparse
from pathlib import Path
from llama_cpp import Llama
from loader import build_corpus
from embedder import load_or_compute_embeddings
from retriever import get_context

def build_prompt(context: str, query: str) -> str:
    return (
        "Répond professionnellement uniquement avec les informations présentes dans le contexte ci-dessous.\n\n"
        f"Contexte:\n{context}\n\n"
        f"Question: {query}\nRéponse:"
    )

def main(model_path: str, data_dir: str, embed_model_name: str, max_tokens: int, temperature: float, threshold: float):
    print("Chargement du corpus (lecture + chunking)...")
    documents, doc_ids = build_corpus(data_dir)
    print(f"{len(documents)} morceaux de texte chargés.")

    print("Chargement / calcul des embeddings...")
    doc_embeddings, embed_model = load_or_compute_embeddings(documents, model_name=embed_model_name)

    print("Initialisation du modèle LLaMA...")
    llm = Llama(model_path=model_path, n_threads=4)

    print("Prêt. Tapez 'exit' ou 'quit' pour quitter.")
    while True:
        q = input("Question: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        context, sources = get_context(q, embed_model, doc_embeddings, documents, doc_ids, top_k=6, threshold=threshold)
        if context is None:
            print("Je n'ai pas d'information fiable pour répondre à cette question.")
            continue

        prompt = build_prompt(context, q)
        output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["\nQuestion:"])
        answer = output["choices"][0]["text"].strip()

        print("Réponse de l'IA:")
        print(answer)
        print("Source(s) utilisée(s):", ", ".join(sources))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../../models/Llama-3.2-1B-Instruct-Q5_K_M.gguf')
    parser.add_argument('--data', type=str, default='../../data')
    parser.add_argument('--embed-model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    main(args.model, args.data, args.embed_model, args.max_tokens, args.temperature, args.threshold)