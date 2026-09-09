# Financial Document RAG Assistant

A retrieval-augmented generation (RAG) system that answers natural-language questions about bank 10-K filings, grounded in the source documents, with a built-in evaluation framework measuring retrieval quality and answer grounding.

## What it does

Ask plain-English questions about the annual reports (10-Ks) of JPMorgan Chase, Bank of America, and Wells Fargo, and get answers grounded in the actual filings, with citations to the source bank.

**Example:**

> **Q:** How do these banks describe their approach to credit risk?
>
> **A:** *(A synthesized, side-by-side answer citing JPMorgan and Bank of America's actual risk-management language: credit risk definitions, concentration limits, mitigation tools like syndications and credit derivatives, and more.)*

## Architecture

The pipeline has five stages:

1. **Document ingestion:** `download_filings.py` programmatically pulls the latest 10-K for each bank directly from the SEC EDGAR system using company CIK identifiers.
2. **Text extraction and chunking:** HTML filings are stripped to clean text and split into ~500-word overlapping chunks.
3. **Embedding:** each chunk is embedded into a vector using a local `sentence-transformers` model (`all-MiniLM-L6-v2`).
4. **Retrieval:** an incoming question is embedded and matched against all chunks via cosine similarity; the top-k most relevant chunks are retrieved.
5. **Generation:** the retrieved chunks and question are passed to a GPT model deployed on Azure AI Foundry, which produces an answer grounded strictly in the retrieved context, with source citations.

### Design decision: hybrid embedding and generation

Generation uses a GPT model deployed on **Azure AI Foundry**. For embeddings, I use a **local model** rather than a cloud embedding endpoint. This was a deliberate choice: after the Azure resource's embedding route proved incompatible with the available API versions, switching to local embeddings removed an external dependency, eliminated per-call cost and latency for retrieval, and made the retrieval layer fully reproducible offline. Cloud is used where it adds the most value (generation); local is used where it is simpler and free (embedding).

## Evaluation

Unlike a basic RAG demo, this project includes an evaluation framework (`evaluate.py`) that measures system quality against known-correct test cases:

| Metric | Result |
|---|---|
| Retrieval accuracy (expected source in top 5) | 100% |
| Top-1 retrieval accuracy | 80% |
| Answers citing the expected source (grounding) | 100% |

**Finding:** The one top-1 retrieval miss occurred on a question about derivative credit exposure, where JPMorgan and Bank of America use closely similar language. The retriever surfaced a semantically near-identical JPMorgan chunk. This illustrates a real RAG challenge: distinguishing between highly similar passages across different source documents.

## Tech stack

Python · sentence-transformers (local embeddings) · Azure AI Foundry (GPT generation) · NumPy (vector similarity) · BeautifulSoup (HTML parsing) · SEC EDGAR API

## How to run

```bash
pip install -r requirements.txt
python download_filings.py   # pull the 10-Ks from SEC EDGAR
python build_rag.py          # extract, chunk, and embed
python ask.py                # ask questions interactively
python evaluate.py           # run the evaluation suite
streamlit run app.py         # launch the conversational web interface
```

Requires a `.env` file with Azure credentials (see `.env.example` for the required variables).

## Web interface

`app.py` provides a Streamlit chat interface with conversation memory. The assistant remembers prior turns, so follow-up questions like "what about Bank of America?" are understood in context.

```bash
streamlit run app.py
```

### Known limitation

For figures that appear in multiple places within a filing (for example, income tax expense in both the income statement and the tax footnote), the chunks retrieved can vary between conversational turns, occasionally surfacing slightly different values. Addressing this in production would involve re-ranking retrieved results or having the model reconcile figures across multiple sources.