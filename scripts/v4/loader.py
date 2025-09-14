from pathlib import Path
from typing import List, Tuple

def read_text_files(data_dir: str = "../../data") -> Tuple[List[str], List[str]]:
    data_dir = Path(data_dir)
    files = sorted([p for p in data_dir.glob("*.txt") if p.is_file()])
    docs = [p.read_text(encoding="utf-8") for p in files]
    ids = [p.name for p in files]
    return docs, ids

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Découpe le texte en chunks basés sur les mots."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

def build_corpus(data_dir: str = "../../data", chunk_size: int = 500, overlap: int = 100):
    raw_docs, raw_ids = read_text_files(data_dir)
    documents = []
    doc_ids = []
    for text, fid in zip(raw_docs, raw_ids):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            documents.append(c)
            doc_ids.append(f"{fid}::chunk_{i}")
    return documents, doc_ids