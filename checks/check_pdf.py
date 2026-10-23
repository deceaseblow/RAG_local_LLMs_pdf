import glob
from pypdf import PdfReader

for path in glob.glob("data/*.pdf"):
    reader = PdfReader(path)
    texts = [(p.extract_text() or "").strip() for p in reader.pages]
    empty = sum(1 for t in texts if len(t) < 50)
    print(f"{path}: {len(texts)} pages, {empty} with almost no text")
    for i, t in enumerate(texts):
        if len(t) >= 50:
            print(f"--- sample from page {i} ---")
            print(t[:600])
            break