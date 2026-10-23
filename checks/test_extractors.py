import argparse
import glob
import random
import time

import numpy as np
import pymupdf  

# ttesting out different OCR engines on a few random pages of a PDF. This is useful for comparing accuracy and speed. The engines are:

DATA_PATH = "data"
ENGINES = ["pypdf", "rapidocr", "tesseract", "winocr", "easyocr", "paddleocr", "ollama"]

def page_image(page, dpi):
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


def make_extractor(engine, args):
    if engine == "pypdf":
        # Embedded text layer only, no OCR. Baseline.
        def run(page, dpi):
            return page.get_text()
        return run

    if engine == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

        def run(page, dpi):
            result, _ = ocr(page_image(page, dpi))
            if not result:
                return ""
            result = sorted(result, key=lambda r: (round(r[0][0][1] / 25), r[0][0][0]))
            return "\n".join(r[1] for r in result)
        return run

    if engine == "tesseract":
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path

        def run(page, dpi):
            img = Image.fromarray(page_image(page, dpi))
            return pytesseract.image_to_string(img, config=f"--psm {args.psm}")
        return run

    if engine == "winocr":
        # Windows built-in OCR. pip install winocr
        import winocr
        from PIL import Image

        def run(page, dpi):
            img = Image.fromarray(page_image(page, dpi))
            result = winocr.recognize_pil_sync(img, args.lang)
            if isinstance(result, dict):
                return result.get("text", "")
            return getattr(result, "text", "")
        return run

    if engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)

        def run(page, dpi):
            lines = reader.readtext(page_image(page, dpi), detail=0, paragraph=True)
            return "\n".join(lines)
        return run

    if engine == "paddleocr":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="en")

        def run(page, dpi):
            img = page_image(page, dpi)
            if hasattr(ocr, "predict"):
                out = ocr.predict(img)
                texts = []
                for item in out:
                    texts.extend(item.get("rec_texts", []))
                return "\n".join(texts)
            out = ocr.ocr(img)
            lines = []
            for block in out or []:
                for entry in block or []:
                    lines.append(entry[1][0])
            return "\n".join(lines)
        return run

    if engine == "ollama":
        # ollama pull qwen2.5vl:7b
        import ollama

        def run(page, dpi):
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            resp = ollama.chat(
                model=args.vision_model,
                messages=[{
                    "role": "user",
                    "content": "Transcribe all the text on this page exactly as written, "
                               "in reading order. Output only the text.",
                    "images": [png],
                }],
                options={"temperature": 0},
            )
            return resp["message"]["content"]
        return run

    raise SystemExit(f"Unknown engine: {engine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=ENGINES)
    parser.add_argument("--n", type=int, default=5, help="how many random pages")
    parser.add_argument("--pages", type=int, nargs="*", help="specific 1-based pages instead of random")
    parser.add_argument("--seed", type=int, default=None, help="same seed = same random pages")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--psm", type=int, default=3, help="tesseract page segmentation mode (3, 6, 11)")
    parser.add_argument("--lang", default="en", help="winocr language tag")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b")
    parser.add_argument("--tesseract-path", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    parser.add_argument("--save", action="store_true", help="also write output to test_<engine>.txt")
    args = parser.parse_args()

    pdf_path = sorted(glob.glob(f"{DATA_PATH}/*.pdf"))[0]
    doc = pymupdf.open(pdf_path)
    print(f"PDF: {pdf_path} ({len(doc)} pages)")

    if args.pages:
        chosen = [p - 1 for p in args.pages]
    else:
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(range(len(doc)), min(args.n, len(doc))))
    print(f"Engine: {args.engine} | DPI: {args.dpi} | pages: {[p + 1 for p in chosen]}\n")

    extract = make_extractor(args.engine, args)
    report = []

    for idx in chosen:
        start = time.time()
        text = extract(doc[idx], args.dpi).strip()
        secs = time.time() - start
        block = f"===== PAGE {idx + 1} | {len(text)} chars | {secs:.1f}s =====\n{text}\n"
        print(block)
        report.append(block)

    if args.save:
        out = f"test_{args.engine}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"saved {out}")


if __name__ == "__main__":
    main()