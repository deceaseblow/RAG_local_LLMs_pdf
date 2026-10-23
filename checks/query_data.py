import argparse
import re

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
LLM_MODEL = "llama3.1:8b"
TOP_K = 12
CANDIDATES = 20
RRF_K = 60

PROMPT_TEMPLATE = """
You are answering questions about a novel using excerpts from it.

{context}

---

Question: {question}

Instructions: Combine clues from the different excerpts above and give the most
likely answer, even if no single sentence states it directly. Give the answer in
one sentence first. Then, only if an excerpt clearly supports it, add one short
quoted phrase from that excerpt as evidence. Do not describe events that are not
in the excerpts. Only say you don't know if the excerpts contain nothing relevant
to the question.
"""

EXPAND_PROMPT = """You help search a long document. Given a question, write 3 short
keyword-style search queries that would match the passages answering it. Use
character names and the words the document itself would likely use (events,
places, verbs like engaged, eloped, accepted, married), not the question's wording.
Output only the 3 queries, one per line, no numbering, no extra text.

Question: {question}
"""

_cache = {}


def make_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0, num_ctx=8192)


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def get_bm25(db):
    if "bm25" not in _cache:
        data = db.get(include=["documents", "metadatas"])
        docs = []
        for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            meta = dict(meta or {})
            meta["id"] = doc_id
            docs.append(Document(page_content=text, metadata=meta))
        _cache["docs"] = docs
        _cache["bm25"] = BM25Okapi([tokenize(d.page_content) for d in docs])
    return _cache["bm25"], _cache["docs"]


def expand_queries(question: str):
    raw = make_llm().invoke(EXPAND_PROMPT.format(question=question))
    lines = [l.strip(" -*0123456789.\t") for l in raw.splitlines()]
    extra = [l for l in lines if len(l) > 3][:3]
    queries = [question] + extra
    print(f"Search queries: {queries}")
    return queries


def hybrid_search(db, queries, k: int = TOP_K):
    bm25, docs = get_bm25(db)
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

    best_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [by_id[i] for i in best_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    queries = expand_queries(query_text)
    results = hybrid_search(db, queries, k=TOP_K)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = make_llm().invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text


if __name__ == "__main__":
    main()