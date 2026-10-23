import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import time
import traceback

import numpy as np
import pymupdf

PAGE = 36  # 1-based page number
DPI = 300
DATA_PATH = "data"
OUT_DIR = "page36_results"
DARIA_PDF = os.path.join(DATA_PATH, "pdfcoffee.com_the-daria-diaries-ordartzpdf-pdf-free.pdf")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
VISION_MODEL = "qwen2.5vl:7b"
VELLUM_TIMEOUT = 600  # seconds


def find_pdf(cli_path):
    """The Daria Diaries PDF unless --pdf is given. Never guesses an unrelated PDF."""
    if cli_path:
        if not os.path.exists(cli_path):
            raise SystemExit(f"PDF not found: {cli_path}")
        return cli_path
    if os.path.exists(DARIA_PDF):
        return DARIA_PDF
    matches = [
        p for p in glob.glob(os.path.join(DATA_PATH, "*.pdf"))
        if "daria" in os.path.basename(p).lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Several PDFs match 'daria': {matches}. Pick one with --pdf.")
    raise SystemExit(f"Could not find the Daria Diaries PDF in {DATA_PATH}/. Pass its path with --pdf.")


def page_image(page, dpi=DPI):
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


def run_pypdf(page, pdf_path):
    # Embedded text layer only. Should be empty for a scanned PDF.
    return page.get_text()


def run_rapidocr(page, pdf_path):
    from rapidocr_onnxruntime import RapidOCR
    result, _ = RapidOCR()(page_image(page))
    if not result:
        return ""
    result = sorted(result, key=lambda r: (round(r[0][0][1] / 25), r[0][0][0]))
    return "\n".join(r[1] for r in result)


def make_tesseract(psm):
    def run(page, pdf_path):
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        img = Image.fromarray(page_image(page))
        return pytesseract.image_to_string(img, config=f"--psm {psm}")
    return run


def run_winocr(page, pdf_path):
    import winocr
    from PIL import Image
    result = winocr.recognize_pil_sync(Image.fromarray(page_image(page)), "en")
    if isinstance(result, dict):
        return result.get("text", "")
    return getattr(result, "text", "")


def run_easyocr(page, pdf_path):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    return "\n".join(reader.readtext(page_image(page), detail=0, paragraph=True))


def run_ollama(page, pdf_path):
    import ollama
    png = page.get_pixmap(dpi=DPI).tobytes("png")
    resp = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": "Transcribe all the text on this page exactly as written, "
                       "in reading order. Output only the text.",
            "images": [png],
        }],
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def run_chrome(page, pdf_path):
    # Chrome's Screen AI OCR through the vellum command line tool.
    vellum = shutil.which("vellum") or os.path.join(
        os.environ.get("USERPROFILE", ""), ".dotnet", "tools", "vellum.exe"
    )
    print("    (Chrome OCR loads its models first, this can take a minute or two)", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [vellum, "ocr", os.path.abspath(pdf_path), "-p", str(PAGE), "-o", tmp]
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=VELLUM_TIMEOUT
        )
        if result.returncode != 0:
            raise RuntimeError(f"vellum exit code {result.returncode}\n{result.stderr or result.stdout}")
        files = glob.glob(os.path.join(tmp, "*_ocr.txt"))
        if not files:
            raise RuntimeError("vellum finished but wrote no text file")
        with open(files[0], encoding="utf-8") as f:
            return f.read()


ENGINES = {
    "pypdf": run_pypdf,
    "chrome_ocr": run_chrome,
    "tesseract_psm3": make_tesseract(3),
    "tesseract_psm11": make_tesseract(11),
    "winocr": run_winocr,
    "rapidocr": run_rapidocr,
    "easyocr": run_easyocr,
    "ollama_vision": run_ollama,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(ENGINES), help="run just these engines")
    parser.add_argument("--pdf", default=None, help="PDF path (default: the Daria Diaries PDF in data/)")
    args = parser.parse_args()

    pdf_path = find_pdf(args.pdf)
    doc = pymupdf.open(pdf_path)
    if PAGE > len(doc):
        raise SystemExit(f"{pdf_path} has only {len(doc)} pages")
    page = doc[PAGE - 1]

    os.makedirs(OUT_DIR, exist_ok=True)
    page.get_pixmap(dpi=130).save(os.path.join(OUT_DIR, f"page{PAGE}.png"))
    print(f"PDF: {pdf_path} ({len(doc)} pages)")
    print(f"Page {PAGE} | DPI {DPI} | results folder: {OUT_DIR}\n")

    summary = []
    for name in (args.only or ENGINES):
        print(f"--- {name} ...", flush=True)
        for stale in (f"{name}.txt", f"{name}.error.txt"):
            path = os.path.join(OUT_DIR, stale)
            if os.path.exists(path):
                os.remove(path)

        started = time.time()
        try:
            text = ENGINES[name](page, pdf_path).strip()
            secs = time.time() - started
            with open(os.path.join(OUT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            line = f"{name:16} ok      {len(text):6} chars  {secs:6.1f}s"
        except Exception as e:
            secs = time.time() - started
            with open(os.path.join(OUT_DIR, f"{name}.error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            line = f"{name:16} FAILED  {type(e).__name__}: {str(e)[:70]!r}  ({secs:.1f}s)"
        print(line + "\n", flush=True)
        summary.append(line)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"page {PAGE} of {os.path.basename(pdf_path)}, DPI {DPI}\n\n" + "\n".join(summary) + "\n")
    print("=== Summary ===")
    print("\n".join(summary))


if __name__ == "__main__":
    main()