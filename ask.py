from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import pickle
import os

load_dotenv()

# Load the saved chunks + embeddings
with open("rag_data.pkl", "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
sources = data["sources"]
embeddings = data["embeddings"]

# Local embedder (same model we built the index with)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Azure client for the CHAT model (this part works)
client = OpenAI(
    base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

def retrieve(question, top_k=5):
    """Find the top_k most relevant chunks for a question."""
    q_vec = embedder.encode([question])[0]
    # cosine similarity between question and every chunk
    sims = embeddings @ q_vec / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    )
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], sources[i], sims[i]) for i in top_idx]

def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src, score in hits
    )
    prompt = (
        "You are a financial analyst assistant. Answer the question using ONLY the "
        "context below from bank 10-K filings. Cite which bank each fact comes from. "
        "If the answer isn't in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    response = client.responses.create(model=CHAT_MODEL, input=prompt)
    return response.output_text, hits

if __name__ == "__main__":
    print("Financial Document RAG Assistant")
    print("Ask questions about JPMorgan, Bank of America, or Wells Fargo 10-K filings.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        ans, hits = answer(question)
        print("\nANSWER:")
        print(ans)
        print("\n--- Sources retrieved: ---")
        for chunk, src, score in hits:
            print(f"  {src} (similarity: {score:.3f})")
        print()