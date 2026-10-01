# News Research RAG

Ask questions about news articles and get answers **with citations**, built on
retrieval-augmented generation (RAG).

Paste article URLs, click Process, then ask questions in plain English. Answers
come only from those articles — and the app tells you which article each answer
came from, or says plainly when the answer isn't there.

Runs entirely on **free tiers**: Google Gemini for the model and embeddings,
FAISS for local vector search.

---

## What it does

**Indexing** (once per set of URLs)

1. **Load** — fetch each page, strip HTML down to readable text
2. **Split** — cut into ~1000-character chunks with overlap
3. **Embed** — turn each chunk into a 3072-number vector via Gemini
4. **Index** — store in FAISS and save to disk

**Answering** (per question)

5. **Embed** the question the same way
6. **Retrieve** the `k` nearest chunks
7. **Generate** an answer from those chunks only
8. **Cite** the source URLs actually used

---

## Setup

```bash
git clone https://github.com/<you>/news-research-rag.git
cd news-research-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then paste your key into .env
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>.

---

## Layout

| File | Purpose |
| --- | --- |
| `rag.py` | The pipeline — load, split, embed, index, answer. No UI. |
| `app.py` | Streamlit interface. |
| `notebooks/` | Four notebooks building the pipeline step by step. |

The split is deliberate: `rag.py` has no Streamlit import, so the whole pipeline
can be tested without launching a browser.

### Notebooks

1. **`01_load_and_split`** — scraping and chunking; why overlap is a maximum, not a guarantee
2. **`02_embeddings`** — vectors, cosine similarity, and handling the rate limit
3. **`03_faiss`** — building and persisting the index; why FAISS scores mean the *opposite* of cosine
4. **`04_retrieval_chain`** — the full chain, built around diagnosing a real retrieval failure

---

## Free-tier limits

| Operation | Limit | What it affects |
| --- | --- | --- |
| Embedding | 100 texts/minute | Indexing — handled automatically with backoff |
| Chat | 20 requests/day **per model** | Asking questions |

Quotas are **per model**, so if chat runs out, switch models in `.env`:

```
CHAT_MODEL=gemini-3.7-flash
```

Indexing a long article can mean several automatic pauses. That's the backoff
working, not a hang.

---

## Notes

**Tuning `k`.** The sidebar slider controls how many chunks the model sees. Too
few and a relevant passage never reaches it; too many and the prompt gets noisy
and expensive. 6–8 is a reasonable default.

**When an answer looks wrong,** open *"Show the passages retrieved"* first. Most
RAG failures are retrieval failures — the model answered honestly from text that
didn't contain the answer.

**Pickle safety.** `FAISS.load_local` needs
`allow_dangerous_deserialization=True` because the docstore is pickled, and
unpickling runs arbitrary code. That's safe here only because the app loads an
index it wrote itself. Never load a `.pkl` index from an untrusted source.

**`langchain-community`** is being sunset but still hosts FAISS with no
maintained standalone replacement yet. Expect that import to move.

---

## Origin

A rebuild of the [codebasics](https://github.com/codebasics/langchain) news
research tool, which targeted LangChain 0.0.284 and paid OpenAI APIs. Most of
its imports no longer exist — `langchain.chains` was removed entirely in
LangChain 1.x — so the pipeline was rewritten with LCEL and free Google models.
