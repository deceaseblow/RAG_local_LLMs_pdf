import glob
import json
import os

import fitz  # pymupdf
import numpy as np
from rapidocr_onnxruntime import RapidOCR

DATA_PATH = "data"
OCR_PATH = "data_ocr"
DPI = 200
MIN_CHARS = 50


def page_to_text(engine, page):
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    result, _ = engine(img)
    if not result:
        return ""
    # Sort roughly top-to-bottom, then left-to-right.
    result = sorted(result, key=lambda r: (round(r[0][0][1] / 25), r[0][0][0]))
    return "\n".join(r[1] for r in result)


def write_txt(pages, txt_path):
    weak = []
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in pages:
            text = p["text"].strip()
            f.write(f"\n===== PAGE {p['page'] + 1} ({len(text)} chars) =====\n")
            if len(text) < MIN_CHARS:
                weak.append(p["page"] + 1)
                f.write("[almost no text recognized on this page]\n")
            f.write(text + "\n")
    print(f"saved {txt_path}")
    if weak:
        print(f"pages with almost no text ({len(weak)}): {weak}")
    else:
        print("all pages have text")


def main():
    os.makedirs(OCR_PATH, exist_ok=True)
    engine = None

    for pdf_path in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = os.path.join(OCR_PATH, name + ".json")
        txt_path = os.path.join(OCR_PATH, name + ".txt")

        if os.path.exists(json_path):
            print(f"OCR already done for {name}, rebuilding text file")
            with open(json_path, encoding="utf-8") as f:
                pages = json.load(f)["pages"]
            write_txt(pages, txt_path)
            continue

        if engine is None:
            engine = RapidOCR()

        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page_to_text(engine, page)
            pages.append({"page": i, "text": text})
            print(f"{name}: page {i + 1}/{len(doc)} -> {len(text)} chars", flush=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"source": pdf_path, "pages": pages}, f, ensure_ascii=False)
        print(f"saved {json_path}")
        write_txt(pages, txt_path)


if __name__ == "__main__":
    main()