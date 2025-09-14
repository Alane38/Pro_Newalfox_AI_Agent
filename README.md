# Pro NEwalfox AI Agent

## Description

Ce projet est un agent intelligent capable de répondre à des questions en utilisant des documents et des modèles de langage.
Il combine des **embeddings de documents** et un **modèle LLaMA** afin de fournir des réponses précises et pertinentes.

## Structure du Projet

* `data/` : Contient les documents et les données utilisés par l’agent.
* `models/` : Contient les modèles pré-entraînés (non inclus dans le dépôt).
* `scripts/` : Contient les scripts Python pour l’agent.

## Prérequis

* **Python** : version 3.11.7 (recommandée)

## Configuration de l’Environnement Python

Pour éviter les conflits de dépendances, il est recommandé d’utiliser un environnement virtuel Python.

1. **Créer un environnement virtuel** :

   ```sh
   python -m venv .venv
   ```

2. **Activer l’environnement virtuel** :

   * Sur **Windows** :

     ```sh
     .venv\Scripts\activate
     ```
   * Sur **macOS / Linux** :

     ```sh
     source .venv/bin/activate
     ```

3. **Installer les dépendances** :

   ```sh
   pip install -r requirements.txt
   ```

## Modèles Requis

Avant de lancer le projet, vous devez télécharger les modèles suivants et les placer dans le dossier `models/`.

| Nom du fichier                       | Description                                 | Lien de téléchargement |
| ------------------------------------ | ------------------------------------------- | ---------------------- |
| `Llama-3.2-1B-Instruct-Q4_0.gguf`    | Modèle LLaMA 3.2 1B Instruct quantisé en Q4 | [Télécharger](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_4_4.gguf?download=true)       |
| `Llama-3.2-1B-Instruct-Q5_K_M.gguf`  | Variante quantisée en Q5                    | [Télécharger](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf?download=true)       |
| `Llama-3.2-3B.Q5_K_M.gguf`           | Modèle LLaMA 3.2 3B quantisé Q5             | [Télécharger](https://huggingface.co/QuantFactory/Llama-3.2-3B-GGUF/resolve/main/Llama-3.2-3B.Q5_K_M.gguf?download=true)       |
| `llama-7b.Q4_0.gguf`                 | Ancien modèle LLaMA 7B quantisé Q4          | [Télécharger](https://huggingface.co/TheBloke/LLaMA-7b-GGUF/resolve/main/llama-7b.Q4_0.gguf?download=true)       |
| `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` | TinyLLaMA 1.1B optimisé pour le chat, Q8    | [Télécharger](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true)       |

### Recommandations du meilleure modèle testé

- `Llama-3.2-1B-Instruct-Q5_K_M.gguf` : Modèle LLaMA 3.2 1B Instruct quantisé en Q5 (plus rapide et efficace). | [Télécharger](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf?download=true)

## Installation

1. **Cloner le dépôt** :

   ```sh
   git clone https://github.com/votre-utilisateur/votre-depot.git
   cd votre-depot
   ```

2. **Télécharger les modèles listés ci-dessus** et les placer dans `models/`.

3. **Installer les dépendances Python** :

   ```sh
   pip install -r requirements.txt
   ```

4. **Lancer l’agent** :

   ```sh
   python scripts/agent.py
   ```

## Utilisation

Lancez le script principal puis posez vos questions.
L’agent générera une réponse basée sur les documents et modèles disponibles.

```sh
python scripts/agent.py
```

## Contribution

Les contributions sont les bienvenues.
Merci d’ouvrir une *issue* ou une *pull request* pour proposer des améliorations ou corrections.

## Licence

Ce projet est sous licence **MIT**.
Voir le fichier `LICENSE` pour plus de détails.