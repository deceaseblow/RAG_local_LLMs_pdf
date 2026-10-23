import glob
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pymupdf

DATA_PATH = "data"
OUT_PATH = "data_text"
WORKERS = 4
VELLUM = shutil.which("vellum") or os.path.join(
    os.environ["USERPROFILE"], ".dotnet", "tools", "vellum.exe"
)

### using chrome ocr on web to read text and save it to a text file. This is a workaround for scanned PDFs that don't have an embedded text layer. It uses the Chrome PDF viewer's built-in OCR capabilities, which are surprisingly good.

def run_part(pdf, part, start, end):
    part_dir = os.path.join(OUT_PATH, f"_part{part}")
    os.makedirs(part_dir, exist_ok=True)
    subprocess.run(
        [VELLUM, "ocr", pdf, "-p", f"{start}-{end}", "-o", part_dir],
        check=True,
    )
    name = os.path.splitext(os.path.basename(pdf))[0]
    with open(os.path.join(part_dir, name + "_ocr.txt"), encoding="utf-8") as f:
        return part, f.read()


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    for pdf in sorted(glob.glob(os.path.join(DATA_PATH, "*.pdf"))):
        total = len(pymupdf.open(pdf))
        size = -(-total // WORKERS)  # ceiling division
        ranges = [
            (i, i * size + 1, min((i + 1) * size, total))
            for i in range(WORKERS)
            if i * size + 1 <= total
        ]
        print(f"{pdf}: {total} pages, ranges {[(s, e) for _, s, e in ranges]}")

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(lambda r: run_part(pdf, *r), ranges))

        results.sort()
        merged = "\f".join(text.strip("\f") for _, text in results)
        name = os.path.splitext(os.path.basename(pdf))[0]
        out = os.path.join(OUT_PATH, name + "_ocr.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"saved {out}")


if __name__ == "__main__":
    main()