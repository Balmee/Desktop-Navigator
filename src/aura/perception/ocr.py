
import os
import shutil

import pytesseract
from PIL import Image


def configure_tesseract():
    """Find the local Tesseract executable."""
    configured_path = os.getenv("TESSERACT_CMD")

    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path
        return

    executable = shutil.which("tesseract")

    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
        return

    common_path = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.path.exists(common_path):
        pytesseract.pytesseract.tesseract_cmd = common_path
        return

    raise RuntimeError(
        "Tesseract was not found. Install it or set "
        "the TESSERACT_CMD environment variable."
    )


def extract_text(
    image: Image.Image,
    confidence_threshold: int = 45,
    max_lines: int = 60,
) -> dict:
    """
    Extract visible text from an image.
    The image is processed in memory.
    """
    configure_tesseract()

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 11",
    )

    words = []

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            continue

        if confidence < confidence_threshold:
            continue

        words.append({
            "text": text,
            "confidence": round(confidence, 1),
            "block": data["block_num"][index],
            "paragraph": data["par_num"][index],
            "line": data["line_num"][index],
        })

    grouped_lines = {}

    for word in words:
        key = (
            word["block"],
            word["paragraph"],
            word["line"],
        )

        grouped_lines.setdefault(key, []).append(word["text"])

    lines = [
        " ".join(parts)
        for parts in grouped_lines.values()
    ][:max_lines]

    return {
        "lines": lines,
        "word_count": len(words),
        "line_count": len(lines),
    }
