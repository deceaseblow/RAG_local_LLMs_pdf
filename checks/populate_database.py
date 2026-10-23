import argparse
import os
import shutil
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"
DATA_PATH = "data"
BATCH_SIZE = 25
MAX_RETRIES = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding new documents: {len(new_chunks)}")
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        batch_ids = [c.metadata["id"] for c in batch]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch, ids=batch_ids)
                break
            except Exception as e:
                print(f"   batch starting at {i} failed (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(5)

        done = min(i + BATCH_SIZE, len(new_chunks))
        print(f"   embedded {done}/{len(new_chunks)}", flush=True)

    print("✅ Done")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()