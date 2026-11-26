
from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract

from PIL import Image

from aura.perception.ocr import configure_tesseract


class NumericOCR:
    """
    Specialised OCR for numerical gameplay HUD elements.
    """

    def __init__(self):
        configure_tesseract()

    
    def preprocess(
        self,
        image: Image.Image,
    ) -> dict[str, np.ndarray]:

        rgb = np.array(image.convert("RGB"))

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=6,
            fy=6,
            interpolation=cv2.INTER_CUBIC,
        )

        variants = {}

        # Test multiple deskew angles.
        for angle in (0, -8, -12, -16):

            height, width = enlarged.shape

            matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                angle,
                1.0,
            )

            rotated = cv2.warpAffine(
                enlarged,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            enhanced = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(rotated)

            _, binary = cv2.threshold(
                enhanced,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            variants[f"gray_{angle}"] = rotated
            variants[f"binary_{angle}"] = binary
            variants[f"inverted_{angle}"] = (
                cv2.bitwise_not(binary)
            )

        return variants


    def extract(
        self,
        image: Image.Image,
    ) -> dict:

        variants = self.preprocess(image)

        candidates = []

        for method, processed in variants.items():

            for psm in (7, 8):

                config = (
                    f"--psm {psm} "
                    "-c tessedit_char_whitelist=0123456789/"
                )

                data = pytesseract.image_to_data(
                    processed,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )

                for text, raw_confidence in zip(
                    data["text"],
                    data["conf"],
                ):

                    text = text.strip()

                    if not re.fullmatch(r"/?\d+", text):
                        continue

                    text = text.lstrip("/")

                    try:
                        confidence = float(raw_confidence)
                    except (ValueError, TypeError):
                        continue

                    if confidence < 0:
                        continue

                    candidates.append({
                        "value": int(text),
                        "confidence": round(confidence, 2),
                        "method": method,
                        "psm": psm,
                    })

        candidates.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "best_candidate": (
                candidates[0] if candidates else None
            ),
            "candidates": candidates,
        }
