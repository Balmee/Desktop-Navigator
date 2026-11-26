
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from aura.perception.health_bar_localiser import (
    HealthBarLocaliser,
)


class HealthBarTracker:
    """
    Experimental multi-scale health HUD tracker.

    Uses colour-segmented reference templates
    to locate the player's health bar.
    """

    SCALES = (
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.35,
    )

    MIN_MATCH_SCORE = 0.55

    def __init__(
        self,
        template_directory: str | Path = (
            "data/templates/health"
        ),
    ):

        self.localiser = HealthBarLocaliser()

        directory = Path(template_directory)

        self.templates = {}

        for name in ("normal", "critical"):

            path = directory / f"{name}_template.png"

            template = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            if template is None:
                raise FileNotFoundError(path)

            self.templates[name] = template

    def locate(self, image: Image.Image) -> dict:

        result = self.localiser.locate(image)

        masks = {
            "normal": np.array(
                result["green_mask"]
            ),
            "critical": np.array(
                result["critical_mask"]
            ),
        }

        candidates = []

        for name, template in self.templates.items():

            search = masks[name]

            for scale in self.SCALES:

                width = round(
                    template.shape[1] * scale
                )

                height = round(
                    template.shape[0] * scale
                )

                if (
                    width > search.shape[1]
                    or height > search.shape[0]
                ):
                    continue

                resized = cv2.resize(
                    template,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                matches = cv2.matchTemplate(
                    search,
                    resized,
                    cv2.TM_CCORR_NORMED,
                )

                _, score, _, position = (
                    cv2.minMaxLoc(matches)
                )

                if not np.isfinite(score):
                    continue

                x, y = position

                candidates.append({
                    "template": name,
                    "score": round(float(score), 4),
                    "scale": scale,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        best = (
            candidates[0]
            if candidates
            else None
        )

        detected = (
            best is not None
            and best["score"] >= self.MIN_MATCH_SCORE
        )

        overlay = np.array(
            result["crop"].convert("RGB")
        )

        if detected:

            x = best["x"]
            y = best["y"]
            width = best["width"]
            height = best["height"]

            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = (
                f"{best['template']} "
                f"{best['score']:.3f} "
                f"x{best['scale']:.2f}"
            )

            cv2.putText(
                overlay,
                label,
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return {
            "detected": detected,
            "best_candidate": best,
            "candidates": candidates[:5],
            "overlay": Image.fromarray(overlay),
        }
