import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"
TEXT_PATH = "data_text"
LOG_PATH = "logs"
BATCH_SIZE = 50
MAX_RETRIES = 3
CANDIDATES = 20
RRF_K = 60
OCR_WORKERS = 4

PROMPT_TEMPLATE = """
You are answering questions about a document using excerpts from it.

{context}

---

Question: {question}

Instructions: Combine clues from the different excerpts above and give the most
likely answer, even if no single sentence states it directly. Give the answer in
one or two sentences first. Then, only if an excerpt clearly supports it, add one
short quoted phrase from that excerpt as evidence. Do not describe events that are
not in the excerpts. Only say you don't know if the excerpts contain nothing
relevant to the question.
"""

EXPAND_PROMPT = """You help search a long document. Given a question, write 3 short
keyword-style search queries that would match the passages answering it. Use
names and the words the document itself would likely use (events, places,
feelings, situations), not the question's wording.
Output only the 3 queries, one per line, no numbering, no extra text.

Question: {question}
"""


# ---------- logging ----------

def _setup_logging():
    os.makedirs(LOG_PATH, exist_ok=True)
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(os.path.join(LOG_PATH, "app.log"), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


log = _setup_logging()
_bm25_cache = {}


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode()


# ---------- paths and library state ----------

def base_name(pdf_name: str) -> str:
    return os.path.splitext(pdf_name)[0]


def collection_name(pdf_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name(pdf_name)).strip("-")[:40] or "doc"
    digest = hashlib.md5(pdf_name.encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


def get_db(coll: str) -> Chroma:
    return Chroma(
        collection_name=coll,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def ocr_text_path(pdf_name: str) -> str:
    return os.path.join(TEXT_PATH, base_name(pdf_name) + "_ocr.txt")


def clipboard_text_path(pdf_name: str) -> str:
    return os.path.join(TEXT_PATH, base_name(pdf_name) + ".txt")


def find_text_path(pdf_name: str):
    """Existing text for this PDF: the vellum file first, then the old clipboard file."""
    for path in (ocr_text_path(pdf_name), clipboard_text_path(pdf_name)):
        if os.path.exists(path) and os.path.getsize(path) > 50:
            return path
    return None


def ready_marker(pdf_name: str) -> str:
    return os.path.join(TEXT_PATH, base_name(pdf_name) + ".ready")


def library_status():
    """(ready, pending): PDFs in data/ that are fully processed vs not yet."""
    ready, pending = [], []
    for path in sorted(glob.glob(os.path.join(DATA_PATH, "*.pdf"))):
        name = os.path.basename(path)
        (ready if os.path.exists(ready_marker(name)) else pending).append(name)
    return ready, pending


def delete_document(pdf_name: str):
    coll = collection_name(pdf_name)
    try:
        get_db(coll).delete_collection()
    except Exception:
        log.exception("Could not delete collection %s", coll)
    _bm25_cache.pop(coll, None)
    candidates = [
        os.path.join(DATA_PATH, pdf_name),
        ocr_text_path(pdf_name),
        clipboard_text_path(pdf_name),
        ocr_text_path(pdf_name).replace("_ocr.txt", "_ocr.json"),
        ready_marker(pdf_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            os.remove(p)
    log.info("Deleted document %s", pdf_name)


# ---------- preflight checks ----------

def find_vellum() -> str:
    found = shutil.which("vellum")
    if found:
        return found
    fallback = os.path.join(os.environ.get("USERPROFILE", ""), ".dotnet", "tools", "vellum.exe")
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError("vellum was not found. Install it with: dotnet tool install -g Ocr.Vellum.Cli")


def preflight(need_vellum: bool):
    if need_vellum:
        log.info("Preflight: vellum found at %s", find_vellum())
    log.info("Preflight: testing the embedding model through Ollama")
    started = time.time()
    try:
        vec = get_embedding_function().embed_query("connection test")
    except Exception as e:
        log.exception("Embedding test failed")
        raise RuntimeError(
            "The Ollama embedding test failed, so nothing was processed. "
            "Check that Ollama is running, that 'ollama pull nomic-embed-text' was done, "
            f"and read the Ollama log (server.log). Original error: {e}"
        ) from e
    log.info("Preflight: embedding model OK (vector size %d, %.1fs)", len(vec), time.time() - started)


# ---------- OCR with vellum (parallel page ranges) ----------

def run_ocr(pdf_name: str, on_part=None) -> str:
    """Chrome OCR through vellum, split into page ranges that run in parallel."""
    os.makedirs(TEXT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    existing = find_text_path(pdf_name)
    if existing:
        log.info("OCR skipped, using existing text file: %s", existing)
        return existing

    base = base_name(pdf_name)
    pdf_path = os.path.abspath(os.path.join(DATA_PATH, pdf_name))
    vellum = find_vellum()

    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
    if total == 0:
        raise RuntimeError("The PDF has no pages")

    workers = max(1, min(OCR_WORKERS, total))
    size = -(-total // workers)  # ceiling division
    ranges = [
        (i, i * size + 1, min((i + 1) * size, total))
        for i in range(workers)
        if i * size + 1 <= total
    ]
    log.info("OCR: %d pages split into %d parts: %s", total, len(ranges), [(s, e) for _, s, e in ranges])

    part_dirs = []

    def run_part(part, start, end):
        part_dir = os.path.abspath(os.path.join(TEXT_PATH, f"_part_{base}_{part}"))
        part_dirs.append(part_dir)
        os.makedirs(part_dir, exist_ok=True)
        cmd = [vellum, "ocr", pdf_path, "-p", f"{start}-{end}", "-o", part_dir]
        started = time.time()
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")

        part_log = os.path.join(LOG_PATH, f"{base}_vellum_part{part}.log")
        with open(part_log, "w", encoding="utf-8") as f:
            f.write(f"command: {' '.join(cmd)}\nexit code: {result.returncode}\n\n")
            f.write("--- stdout ---\n" + (result.stdout or "") + "\n--- stderr ---\n" + (result.stderr or ""))
        log.info("OCR part %d (pages %d-%d) exited with code %s after %.0fs (log: %s)",
                 part, start, end, result.returncode, time.time() - started, part_log)

        if result.returncode != 0:
            tail = _ascii((result.stderr or result.stdout or "")[-500:])
            raise RuntimeError(f"vellum failed on pages {start}-{end} (exit code {result.returncode}). "
                               f"See {part_log}. Last output:\n{tail}")

        part_txt = os.path.join(part_dir, base + "_ocr.txt")
        if not os.path.exists(part_txt):
            found = glob.glob(os.path.join(part_dir, "*_ocr.txt"))
            if len(found) != 1:
                raise RuntimeError(f"vellum finished pages {start}-{end} but no text file appeared. See {part_log}.")
            part_txt = found[0]
        with open(part_txt, encoding="utf-8") as f:
            return part, f.read()

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [pool.submit(run_part, *r) for r in ranges]
            for done, fut in enumerate(as_completed(futures), start=1):
                part, text = fut.result()  # raises here if a part failed
                results[part] = text
                log.info("OCR parts finished: %d/%d", done, len(ranges))
                if on_part:
                    on_part(done, len(ranges))

        merged = "\f".join(results[i].strip("\f") for i in sorted(results))
        pages_found = merged.count("\f") + 1
        if pages_found != total:
            log.warning("PDF has %d pages but the merged OCR text has %d page sections, "
                        "so page numbers in Sources may be off", total, pages_found)

        out = ocr_text_path(pdf_name)
        with open(out, "w", encoding="utf-8") as f:
            f.write(merged)
        size_bytes = os.path.getsize(out)
        log.info("OCR text file: %s (%d bytes)", out, size_bytes)
        if size_bytes < 50:
            raise RuntimeError(f"OCR produced almost no text ({size_bytes} bytes)")
        return out
    finally:
        for d in part_dirs:
            shutil.rmtree(d, ignore_errors=True)


# ---------- chunking and embedding ----------

def load_pages(txt_path: str):
    with open(txt_path, encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("\f")
    docs = []
    for i, text in enumerate(parts):
        if len(text.strip()) < 20:
            continue
        docs.append(Document(page_content=text.strip(), metadata={"source": txt_path, "page": i + 1}))
    log.info("Text split into %d pages, %d with text", len(parts), len(docs))
    if len(parts) == 1:
        log.warning("No page breaks in this file, so all text is treated as page 1")
    return docs


def make_chunks(docs, coll):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    last_page, idx = None, 0
    for chunk in chunks:
        page = chunk.metadata["page"]
        idx = idx + 1 if page == last_page else 0
        chunk.metadata["id"] = f"{coll}:{page}:{idx}"
        last_page = page
    return chunks


def ingest(pdf_name: str, progress=None):
    """Check Ollama, OCR if needed, chunk, embed, verify, mark ready."""
    def report(stage, done=0, total=0):
        log.info("[%s] %s%s", pdf_name, stage, f" {done}/{total}" if total else "")
        if progress:
            progress(stage, done, total)

    coll = collection_name(pdf_name)
    started = time.time()
    log.info("=== Processing %s (collection %s) ===", pdf_name, coll)

    if not os.path.exists(os.path.join(DATA_PATH, pdf_name)):
        raise FileNotFoundError(f"{pdf_name} is not in the {DATA_PATH} folder")

    report("Step 1/5: checking the Ollama embedding model")
    preflight(need_vellum=find_text_path(pdf_name) is None)

    report("Step 2/5: reading the text with Chrome OCR (a few minutes for long files)")
    txt_path = run_ocr(
        pdf_name,
        on_part=lambda done, total: report("Step 2/5: Chrome OCR, parts finished", done, total),
    )

    report("Step 3/5: splitting into chunks")
    chunks = make_chunks(load_pages(txt_path), coll)
    log.info("Created %d chunks", len(chunks))
    if not chunks:
        raise RuntimeError("The text file produced no chunks (is it empty?)")

    db = get_db(coll)
    existing = set(db.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing]
    log.info("%d chunks total, %d already stored, %d to embed", len(chunks), len(existing), len(new_chunks))

    report("Step 4/5: embedding chunks", 0, len(new_chunks))
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        ids = [c.metadata["id"] for c in batch]
        batch_started = time.time()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch, ids=ids)
                break
            except Exception as e:
                log.warning("Batch starting at %d failed (attempt %d/%d): %r", i, attempt, MAX_RETRIES, e)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(5)
        log.info("Batch %d-%d embedded in %.1fs", i, i + len(batch), time.time() - batch_started)
        report("Step 4/5: embedding chunks", min(i + BATCH_SIZE, len(new_chunks)), len(new_chunks))

    report("Step 5/5: verifying")
    stored = len(db.get(include=[])["ids"])
    if stored < len(chunks):
        raise RuntimeError(f"Only {stored} of {len(chunks)} chunks were stored")

    with open(ready_marker(pdf_name), "w", encoding="utf-8") as f:
        f.write(str(len(chunks)))
    _bm25_cache.pop(coll, None)
    log.info("=== Done: %s ready (%d chunks, %.0fs total) ===", pdf_name, len(chunks), time.time() - started)


# ---------- retrieval and answering ----------

def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def get_bm25(db, coll):
    if coll not in _bm25_cache:
        data = db.get(include=["documents", "metadatas"])
        docs = []
        for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            meta = dict(meta or {})
            meta["id"] = doc_id
            docs.append(Document(page_content=text, metadata=meta))
        if not docs:
            raise RuntimeError("This document has no stored chunks. Delete it and process it again.")
        _bm25_cache[coll] = (BM25Okapi([tokenize(d.page_content) for d in docs]), docs)
        log.info("Built keyword index for %s (%d chunks)", coll, len(docs))
    return _bm25_cache[coll]


def make_llm(model: str, num_ctx: int):
    return OllamaLLM(model=model, temperature=0, num_ctx=num_ctx, keep_alive="30m")


def expand_queries(question: str, model: str, num_ctx: int):
    try:
        raw = make_llm(model, num_ctx).invoke(EXPAND_PROMPT.format(question=question))
        lines = [l.strip(" -*0123456789.\t") for l in raw.splitlines()]
        extra = [l for l in lines if len(l) > 3][:3]
    except Exception as e:
        log.exception("Query expansion failed, searching with the original question only: %r", e)
        extra = []
    return [question] + extra


def hybrid_search(pdf_name: str, queries, k: int):
    coll = collection_name(pdf_name)
    db = get_db(coll)
    bm25, docs = get_bm25(db, coll)
    by_id = {d.metadata["id"]: d for d in docs}
    fused = {}

    for q in queries:
        vector_ids = [d.metadata["id"] for d in db.similarity_search(q, k=CANDIDATES)]
        scores = bm25.get_scores(tokenize(q))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
        bm25_ids = [docs[i].metadata["id"] for i in top_idx]
        for ranking in (vector_ids, bm25_ids):
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [by_id[i] for i in best if i in by_id]


def retrieve(pdf_name: str, question: str, model: str, k: int, expand: bool, num_ctx: int):
    started = time.time()
    queries = expand_queries(question, model, num_ctx) if expand else [question]
    log.info("Search queries: %s", _ascii(str(queries)))
    results = hybrid_search(pdf_name, queries, k)
    log.info("Retrieved %d chunks in %.1fs", len(results), time.time() - started)
    return queries, results


def stream_answer(question: str, results, model: str, num_ctx: int):
    context = "\n\n---\n\n".join(d.page_content for d in results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=question
    )
    log.info("Generating answer with %s (prompt %d characters)", model, len(prompt))
    started = time.time()
    try:
        for piece in make_llm(model, num_ctx).stream(prompt):
            yield piece
    except Exception:
        log.exception("Answer generation failed")
        raise
    log.info("Answer finished in %.1fs", time.time() - started)