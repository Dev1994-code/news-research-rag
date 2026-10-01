"""Streamlit UI for the news research tool.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

import rag

st.set_page_config(page_title="News Research Tool", page_icon="📰")

st.title("📰 News Research Tool")
st.caption("Paste article URLs, then ask questions. Answers cite their source.")

# --------------------------------------------------------------------------- #
# Sidebar: the URLs and the Process button
# --------------------------------------------------------------------------- #
st.sidebar.header("Article URLs")
urls = [
    st.sidebar.text_input(f"URL {i + 1}", key=f"url{i}")
    for i in range(3)
]
process = st.sidebar.button("Process URLs", type="primary")

st.sidebar.divider()
k = st.sidebar.slider(
    "Passages per question (k)", 2, 15, 8,
    help="How many chunks to give the model. Too few and it misses the answer; "
         "too many and the prompt gets noisy and expensive.",
)

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("No GOOGLE_API_KEY found. Put it in rebuild/.env and restart.")
    st.stop()

# The index is cached in session state so questions don't rebuild it.
if "store" not in st.session_state:
    st.session_state.store = rag.load_index()

# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
if process:
    entered = [u for u in urls if u.strip()]
    if not entered:
        st.sidebar.error("Enter at least one URL.")
    else:
        status = st.empty()
        try:
            with st.spinner("Working…"):
                st.session_state.store = rag.build_index(
                    entered, on_progress=lambda msg: status.info(msg)
                )
            status.success(f"Indexed {len(entered)} article(s).")
        except Exception as exc:
            status.empty()
            st.error(f"Failed: {exc}")

# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #
if st.session_state.store is None:
    st.info("Add article URLs in the sidebar and click **Process URLs** to begin.")
    st.stop()

st.success(f"Index ready — {st.session_state.store.index.ntotal} chunks loaded.")

question = st.text_input("Ask a question about the articles:")

if question:
    try:
        with st.spinner("Searching and answering…"):
            result = rag.answer(st.session_state.store, question, k=k)
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            st.error(
                "Daily free-tier chat quota reached (20 requests/day per model). "
                "Try again tomorrow, or set CHAT_MODEL in .env to a different "
                "model — the quota is per model."
            )
        else:
            st.error(f"Failed: {exc}")
    else:
        st.header("Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("Sources")
            for url in result["sources"]:
                st.markdown(f"- {url}")
        else:
            st.caption("No sources cited — the answer was not found in the articles.")

        with st.expander(f"Show the {len(result['chunks'])} passages retrieved"):
            for i, doc in enumerate(result["chunks"], 1):
                st.markdown(f"**[{i}]** {doc.metadata['source']}")
                st.text(doc.page_content[:400] + "…")
