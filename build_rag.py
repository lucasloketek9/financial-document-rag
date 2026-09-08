from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load a small, fast, free embedding model that runs locally
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def read_document(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    docs = {
        "JPMorgan": "documents/jpmorgan_10k.html",
        "Bank of America": "documents/bankofamerica_10k.html",
        "Wells Fargo": "documents/wellsfargo_10k.html",
    }

    all_chunks, all_sources = [], []
    for bank, path in docs.items():
        print(f"Reading {bank}...")
        chunks = chunk_text(read_document(path))
        all_chunks.extend(chunks)
        all_sources.extend([bank] * len(chunks))
        print(f"  {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding all chunks locally (this may take a minute)...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    with open("rag_data.pkl", "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "sources": all_sources,
            "embeddings": np.array(embeddings),
        }, f)

    print(f"\nDone. Saved {len(all_chunks)} chunks with embeddings to rag_data.pkl")