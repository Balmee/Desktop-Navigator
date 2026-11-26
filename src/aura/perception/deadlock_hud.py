
from __future__ import annotations

from PIL import Image, ImageOps, ImageEnhance

from aura.perception.ocr import extract_text


class DeadlockHUD:
    """
    Extracts and processes specific Deadlock HUD regions.

    Region coordinates are normalised:
    (left, top, right, bottom).
    """

    REGIONS = {
        "player_health": (0.20, 0.60, 0.32, 0.87),
        "player_economy": (0.00, 0.79, 0.21, 0.88),
        "player_stats": (0.00, 0.70, 0.16, 0.79),
        "inventory": (0.00, 0.87, 0.21, 1.00),
        "abilities": (0.39, 0.89, 0.61, 1.00),
        "team_portraits": (0.30, 0.00, 0.70, 0.15),
    }

    def crop_region(
        self,
        image: Image.Image,
        region_name: str,
    ) -> Image.Image:

        if region_name not in self.REGIONS:
            raise ValueError(
                f"Unknown HUD region: {region_name}"
            )

        width, height = image.size

        left, top, right, bottom = self.REGIONS[region_name]

        bounds = (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )

        return image.crop(bounds)

    def preprocess(
        self,
        image: Image.Image,
        scale: int = 3,
    ) -> Image.Image:

        # Remove colour information.
        image = ImageOps.grayscale(image)

        # Increase resolution for small HUD text.
        image = image.resize(
            (
                image.width * scale,
                image.height * scale,
            ),
            Image.Resampling.LANCZOS,
        )

        # Improve contrast between text and background.
        image = ImageOps.autocontrast(image)

        image = ImageEnhance.Contrast(image).enhance(1.5)

        # Sharpen character edges.
        image = ImageEnhance.Sharpness(image).enhance(2.0)

        return image

    def extract_region_text(
        self,
        image: Image.Image,
        region_name: str,
    ) -> dict:

        cropped = self.crop_region(image, region_name)
        processed = self.preprocess(cropped)

        result = extract_text(
            processed,
            confidence_threshold=40,
            max_lines=20,
        )

        return {
            "region": region_name,
            "lines": result["lines"],
            "word_count": result["word_count"],
            "line_count": result["line_count"],
        }

    def analyse(self, image: Image.Image) -> dict:

        results = {}

        for region_name in self.REGIONS:
            results[region_name] = self.extract_region_text(
                image,
                region_name,
            )

        return results
