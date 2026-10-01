"""RAG pipeline for the news research tool.

Everything the notebooks built, as reusable functions. The Streamlit app in
app.py is only a UI on top of this — keeping them separate means the pipeline
can be tested without launching a browser.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")  # hides the langchain-community sunset notice

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_google_genai import (  # noqa: E402
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-2"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.5-flash-lite")
INDEX_DIR = Path(__file__).with_name("faiss_index")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; news-research-tool)"}


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_url(url: str) -> Document:
    """Fetch a page and return its readable text as a Document."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        raise ValueError(f"No readable content found at {url}")

    return Document(
        page_content=main.get_text(separator="\n", strip=True),
        metadata={
            "source": url,
            "title": soup.title.get_text(strip=True) if soup.title else url,
        },
    )


# --------------------------------------------------------------------------- #
# 2. Split
# --------------------------------------------------------------------------- #
def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


# --------------------------------------------------------------------------- #
# 3. Embed
# --------------------------------------------------------------------------- #
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def embed_all(texts: list[str], embeddings, batch_size: int = 40, on_progress=None):
    """Embed with backoff. The free tier allows 100 texts per minute."""
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        while True:
            try:
                vectors.extend(embeddings.embed_documents(batch))
                if on_progress:
                    on_progress(len(vectors), len(texts), None)
                break
            except Exception as exc:
                message = str(exc)
                if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                    raise
                match = re.search(r"retry in ([\d.]+)s", message)
                wait = float(match.group(1)) + 2 if match else 35
                if on_progress:
                    on_progress(len(vectors), len(texts), wait)
                time.sleep(wait)

    return vectors


# --------------------------------------------------------------------------- #
# 4. Index
# --------------------------------------------------------------------------- #
def build_index(urls: list[str], on_progress=None) -> FAISS:
    """Load -> split -> embed -> FAISS, then save to disk."""
    def say(msg):
        if on_progress:
            on_progress(msg)

    say("Loading articles…")
    docs = [load_url(u) for u in urls if u.strip()]
    if not docs:
        raise ValueError("No URLs provided")

    say(f"Splitting {len(docs)} articles…")
    chunks = split(docs)

    say(f"Embedding {len(chunks)} chunks (this can take a minute)…")
    embeddings = get_embeddings()
    vectors = embed_all(
        [c.page_content for c in chunks],
        embeddings,
        on_progress=lambda done, total, wait: say(
            f"Rate limit reached — waiting {wait:.0f}s… ({done}/{total})"
            if wait
            else f"Embedded {done}/{total} chunks"
        ),
    )

    say("Building the index…")
    # from_embeddings reuses the vectors we just made rather than re-embedding.
    store = FAISS.from_embeddings(
        text_embeddings=list(zip([c.page_content for c in chunks], vectors)),
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
    )
    store.save_local(str(INDEX_DIR))
    say(f"Done — {len(chunks)} chunks indexed.")
    return store


def load_index() -> FAISS | None:
    """Load a previously saved index, or None if there isn't one."""
    if not (INDEX_DIR / "index.faiss").exists():
        return None
    # Safe: this file is written by build_index above, not downloaded.
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


# --------------------------------------------------------------------------- #
# 5. Answer
# --------------------------------------------------------------------------- #
class CitedAnswer(BaseModel):
    answer: str = Field(description="Answer using only the context. Say so if absent.")
    sources_used: list[int] = Field(description="Numbers of the passages actually used")


PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the numbered passages below.\n"
    "If the passages do not contain the answer, say so plainly.\n"
    "List the numbers of the passages you actually used.\n\n"
    "{context}\n\nQuestion: {question}"
)


def _format(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def answer(store: FAISS, question: str, k: int = 8) -> dict:
    """Retrieve k chunks, answer from them, and report the sources used."""
    docs = store.as_retriever(search_kwargs={"k": k}).invoke(question)

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, max_retries=3)
    chain = PROMPT | llm.with_structured_output(CitedAnswer)
    result = chain.invoke({"context": _format(docs), "question": question})

    sources = sorted(
        {
            docs[i - 1].metadata["source"]
            for i in result.sources_used
            if 1 <= i <= len(docs)
        }
    )
    return {"answer": result.answer, "sources": sources, "chunks": docs}
