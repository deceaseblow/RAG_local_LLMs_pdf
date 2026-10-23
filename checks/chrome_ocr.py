import glob
import os
import subprocess
import time

import pyautogui
import pyperclip

DATA_PATH = "data"
OUT_PATH = "data_text"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

SCROLL_PAUSE = 1.5  # Seconds to wait per screen
MAX_SCREENS = 120   # Safety limit on Page Down presses
FIRST_WAIT = 8      # Seconds for Chrome to open and render


def main():
    os.makedirs(OUT_PATH, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "*.pdf")))

    if not pdfs:
        print(f"No PDF files found in '{DATA_PATH}'.")
        return

    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        out_file = os.path.join(OUT_PATH, name + ".txt")
        url = "file:///" + os.path.abspath(pdf).replace("\\", "/")

        print(f"Opening {name} in Chrome. Do not touch the keyboard or mouse.")

        subprocess.Popen([CHROME, "--new-window", url])
        time.sleep(FIRST_WAIT)

        # Click the middle of the screen so the PDF viewer has focus.
        w, h = pyautogui.size()
        pyautogui.click(w // 2, h // 2)

        # Scroll through the document to allow text rendering or OCR.
        pyautogui.hotkey("ctrl", "home")
        time.sleep(SCROLL_PAUSE)

        for _ in range(MAX_SCREENS):
            pyautogui.press("pagedown")
            time.sleep(SCROLL_PAUSE)

        time.sleep(5)

        # Select everything and copy.
        pyperclip.copy("")
        pyautogui.hotkey("ctrl", "a")
        time.sleep(1)
        pyautogui.hotkey("ctrl", "c")
        time.sleep(2)

        text = pyperclip.paste()

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved {out_file} ({len(text)} characters)")

        # Close the Chrome window.
        pyautogui.hotkey("ctrl", "w")
        time.sleep(2)


if __name__ == "__main__":
    main()