
from __future__ import annotations

import cv2
import numpy as np

from PIL import Image


class HealthBarLocaliser:
    """
    Experimental dynamic localisation of the Deadlock
    player health bar.
    """

    SEARCH_REGION = (0.15, 0.48, 0.37, 0.94)

    def crop(self, image: Image.Image) -> Image.Image:

        left, top, right, bottom = self.SEARCH_REGION

        return image.crop((
            int(left * image.width),
            int(top * image.height),
            int(right * image.width),
            int(bottom * image.height),
        ))

    def locate(self, image: Image.Image) -> dict:

        cropped = self.crop(image)

        rgb = np.array(cropped.convert("RGB"))

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        # Dark green border: normal health.
        green_mask = cv2.inRange(
            hsv,
            np.array([35, 70, 10]),
            np.array([95, 255, 130]),
        )

        # Orange-red border: low health.
        orange_mask = cv2.inRange(
            hsv,
            np.array([0, 150, 120]),
            np.array([22, 255, 255]),
        )

        red_mask = cv2.inRange(
            hsv,
            np.array([170, 150, 120]),
            np.array([179, 255, 255]),
        )

        critical_mask = cv2.bitwise_or(
            orange_mask,
            red_mask,
        )

        kernel = np.ones((5, 5), dtype=np.uint8)

        candidates = []

        for colour, mask in (
            ("normal", green_mask),
            ("critical", critical_mask),
        ):

            cleaned = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
            )

            contours, _ = cv2.findContours(
                cleaned,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < 500:
                    continue

                x, y, width, height = (
                    cv2.boundingRect(contour)
                )

                if width <= 0:
                    continue

                aspect_ratio = height / width

                if height < cropped.height * 0.25:
                    continue

                if aspect_ratio < 1.2:
                    continue

                hull = cv2.convexHull(contour)

                candidates.append({
                    "colour": colour,
                    "area": round(area, 2),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(
                        aspect_ratio, 3
                    ),
                    "hull": hull.reshape(-1, 2).tolist(),
                })

        candidates.sort(
            key=lambda item: item["area"],
            reverse=True,
        )

        overlay = rgb.copy()

        for index, candidate in enumerate(candidates):

            x = candidate["x"]
            y = candidate["y"]
            width = candidate["width"]
            height = candidate["height"]

            points = np.array(
                candidate["hull"],
                dtype=np.int32,
            )

            cv2.polylines(
                overlay,
                [points],
                True,
                (0, 255, 255),
                2,
            )

            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                overlay,
                f"{index}: {candidate['colour']}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return {
            "crop": cropped,
            "green_mask": Image.fromarray(green_mask),
            "critical_mask": Image.fromarray(
                critical_mask
            ),
            "overlay": Image.fromarray(overlay),
            "candidates": candidates,
        }
