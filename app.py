import os
import traceback

import ollama
import streamlit as st

import rag_core as rag

st.set_page_config(page_title="RAG with local LLMs for PDFs", layout="wide")
os.makedirs(rag.DATA_PATH, exist_ok=True)

ss = st.session_state
ss.setdefault("chats", {})
ss.setdefault("processing", False)
ss.setdefault("job", None)

EMBED_HINTS = ("embed", "bge", "minilm", "e5-", "gte-")
PREFERRED_MODEL = "llama3.1:8b"


@st.cache_data(ttl=15, show_spinner=False)
def installed_models():
    """Names of chat-capable models installed in Ollama (embedding models hidden)."""
    try:
        resp = ollama.list()
        items = resp["models"] if isinstance(resp, dict) else resp.models
        names = []
        for m in items:
            name = m["model"] if isinstance(m, dict) and "model" in m else (
                m["name"] if isinstance(m, dict) else getattr(m, "model", None)
            )
            if name:
                names.append(name)
    except Exception:
        return []
    return sorted(n for n in names if not any(h in n.lower() for h in EMBED_HINTS))


def start_upload_job():
    ss.processing = True
    ss.job = {"type": "upload"}


def start_pending_job():
    ss.processing = True
    ss.job = {"type": "pending", "name": ss.get("pending_pick")}


def process_pdf(pdf_name: str, box) -> bool:
    """Run the full pipeline for one PDF in data/. Returns True on success."""
    with box:
        bar = st.progress(0.0)
        status = st.empty()

    def on_progress(stage, done, total):
        status.write(stage)
        if total:
            bar.progress(min(done / total, 1.0))

    try:
        rag.ingest(pdf_name, on_progress)
    except Exception as e:
        rag.log.exception("Processing %s failed", pdf_name)
        ss["last_error"] = (
            f"Processing {pdf_name} failed: {type(e).__name__}: {e}",
            traceback.format_exc(),
        )
        return False

    ss.pop("last_error", None)
    ss["flash"] = f"{pdf_name} is ready."
    ss["want_doc"] = pdf_name
    return True


busy = ss.processing
ready, pending = rag.library_status()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Add a PDF")

    if "flash" in ss:
        st.success(ss.pop("flash"))

    uploaded = st.file_uploader("Drop a PDF here", type=["pdf"], key="uploader", disabled=busy)

    name_taken = (
        uploaded is not None
        and os.path.exists(os.path.join(rag.DATA_PATH, uploaded.name))
    )
    if name_taken and not busy:
        if uploaded.name in ready:
            st.warning(
                f"A document named {uploaded.name} already exists. "
                "To replace it, pick it in 'Document to chat with' below, click "
                "'Delete this document', then process the new file."
            )
        else:
            st.warning(
                f"A file named {uploaded.name} already exists but was not fully processed. "
                "Open 'PDFs not fully processed' below to finish it or delete it, "
                "then process the new file."
            )

    st.button(
        "Processing..." if busy else "Process this PDF",
        type="primary",
        disabled=busy or uploaded is None or name_taken,
        on_click=start_upload_job,
    )
    progress_box = st.container()

    if "last_error" in ss and not busy:
        message, details = ss["last_error"]
        st.error(message)
        if details:
            with st.expander("Technical details (also in logs/app.log)"):
                st.code(details)
        if st.button("Dismiss error"):
            ss.pop("last_error")
            st.rerun()

    if pending:
        with st.expander(f"PDFs not fully processed ({len(pending)})"):
            st.selectbox("PDF", pending, key="pending_pick", disabled=busy)
            st.button("Process it", disabled=busy, on_click=start_pending_job)
            if st.button("Delete it", disabled=busy, key="delete_pending"):
                victim = ss.get("pending_pick")
                if victim:
                    rag.delete_document(victim)
                    ss.chats.pop(victim, None)
                    st.rerun()

    st.divider()
    if ss.get("want_doc") in ready:
        ss["selected_doc"] = ss.pop("want_doc")
    if ss.get("selected_doc") not in ready:
        ss.pop("selected_doc", None)
    selected = (
        st.selectbox("Document to chat with", ready, key="selected_doc", disabled=busy)
        if ready
        else None
    )

    st.header("Settings")

    models = installed_models()
    if models:
        if ss.get("answer_model") not in models:
            ss.pop("answer_model", None)
        default_index = models.index(PREFERRED_MODEL) if PREFERRED_MODEL in models else 0
        model = st.selectbox(
            "Answer model", models, index=default_index, key="answer_model", disabled=busy
        )
    else:
        st.warning("Could not list models from Ollama. Is it running? Type a model name instead.")
        model = st.text_input("Answer model", value=PREFERRED_MODEL, disabled=busy)

    if st.button("Refresh model list", disabled=busy):
        installed_models.clear()
        st.rerun()

    top_k = st.slider("Chunks to retrieve", 4, 16, 10, disabled=busy)
    num_ctx = st.select_slider("Context window", options=[2048, 4096, 8192], value=4096, disabled=busy)
    expand = st.checkbox("Expand the question first (better recall, slower)", value=True, disabled=busy)

    if selected and st.button("Delete this document", disabled=busy):
        rag.delete_document(selected)
        ss.chats.pop(selected, None)
        st.rerun()

# ---------------- main area ----------------
st.title("RAG with Local LLMs, for pdfs")

if busy:
    st.warning("Processing a PDF. Please wait and do not close this tab. The controls are locked until it finishes.")

if not selected:
    st.info("Upload a PDF in the sidebar and click Process. Scanned PDFs are read with Chrome's OCR.")
else:
    history = ss.chats.setdefault(selected, [])
    st.caption(f"Chatting with: {selected}")

    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for page, snippet in msg["sources"]:
                        st.markdown(f"**Page {page}**: {snippet}")

    question = st.chat_input("Ask a question about this document", disabled=busy)
    if question:
        rag.log.info("Question about %s: %s", selected, question)
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document..."):
                    queries, results = rag.retrieve(selected, question, model, top_k, expand, num_ctx)
                answer = st.write_stream(rag.stream_answer(question, results, model, num_ctx))
            except Exception as e:
                rag.log.exception("Question failed")
                st.error(f"{type(e).__name__}: {e}")
                st.code(traceback.format_exc())
                st.stop()

            sources = [
                (d.metadata.get("page", "?"), d.page_content[:220].replace("\n", " ") + "...")
                for d in results
            ]
            with st.expander("Sources"):
                if expand:
                    st.caption(f"Search queries used: {queries}")
                for page, snippet in sources:
                    st.markdown(f"**Page {page}**: {snippet}")

        history.append({"role": "assistant", "content": answer, "sources": sources})

# ---------------- background job (runs last, after the locked UI is drawn) ----------------
if ss.processing:
    job = ss.job or {}
    try:
        name = None
        if job.get("type") == "upload":
            if uploaded is None:
                ss["last_error"] = ("No file was selected.", "")
            elif os.path.exists(os.path.join(rag.DATA_PATH, uploaded.name)):
                ss["last_error"] = (
                    f"A document named {uploaded.name} already exists, so the upload was refused. "
                    "Delete the old one first.",
                    "",
                )
            else:
                name = uploaded.name
                with open(os.path.join(rag.DATA_PATH, name), "wb") as f:
                    f.write(uploaded.getbuffer())
                rag.log.info("Saved upload %s (%d bytes)", name, uploaded.size)
        elif job.get("type") == "pending":
            name = job.get("name")

        if name:
            process_pdf(name, progress_box)
    except Exception as e:
        rag.log.exception("Job crashed")
        ss["last_error"] = (f"{type(e).__name__}: {e}", traceback.format_exc())
    finally:
        ss.processing = False
        ss.job = None
    st.rerun()