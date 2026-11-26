
from __future__ import annotations

import cv2
import numpy as np

from PIL import Image


class HealthBarDetector:
    """
    Geometry-constrained Deadlock health bar segmentation.

    Calibrated provisionally against a 1920x1080
    practice-room screenshot.
    """

    REGION = (0.20, 0.60, 0.32, 0.87)

    REFERENCE_WIDTH = 230
    REFERENCE_HEIGHT = 291

    BAR_POLYGON = [
        (54, 60),
        (95, 46),
        (95, 25),
        (105, 20),
        (140, 276),
        (125, 279),
    ]

    def crop(self, image: Image.Image) -> Image.Image:

        left, top, right, bottom = self.REGION

        return image.crop((
            int(left * image.width),
            int(top * image.height),
            int(right * image.width),
            int(bottom * image.height),
        ))

    def create_geometry_mask(
        self,
        width: int,
        height: int,
    ) -> np.ndarray:

        scale_x = width / self.REFERENCE_WIDTH
        scale_y = height / self.REFERENCE_HEIGHT

        points = np.array([
            (
                round(x * scale_x),
                round(y * scale_y),
            )
            for x, y in self.BAR_POLYGON
        ], dtype=np.int32)

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.fillPoly(mask, [points], 255)

        return mask

    def segment(self, image: Image.Image) -> dict:

        cropped = self.crop(image)

        rgb = np.array(cropped.convert("RGB"))

        height, width = rgb.shape[:2]

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        # Detect the pale health bar fill.
        colour_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 160]),
            np.array([179, 90, 255]),
        )

        # Restrict detection to the health bar geometry.
        geometry_mask = self.create_geometry_mask(
            width,
            height,
        )

        mask = cv2.bitwise_and(
            colour_mask,
            geometry_mask,
        )

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        components = []

        for index in range(1, count):

            x, y, component_width, component_height, area = (
                stats[index].tolist()
            )

            if area < 30:
                continue

            components.append({
                "label": index,
                "x": x,
                "y": y,
                "width": component_width,
                "height": component_height,
                "area": area,
                "aspect_ratio": round(
                    component_height / max(component_width, 1),
                    3,
                ),
            })

        components.sort(
            key=lambda item: item["area"],
            reverse=True,
        )

        foreground_pixels = int(
            cv2.countNonZero(mask)
        )

        geometry_pixels = int(
            cv2.countNonZero(geometry_mask)
        )

        coverage = (
            foreground_pixels / geometry_pixels
            if geometry_pixels > 0
            else 0.0
        )

        overlay = rgb.copy()

        overlay[mask > 0] = (
            overlay[mask > 0] * 0.4
            + np.array([0, 255, 0]) * 0.6
        ).astype(np.uint8)

        return {
            "crop": cropped,
            "mask": Image.fromarray(mask),
            "geometry_mask": Image.fromarray(geometry_mask),
            "colour_mask": Image.fromarray(colour_mask),
            "overlay": Image.fromarray(overlay),
            "components": components,
            "foreground_pixels": foreground_pixels,
            "geometry_pixels": geometry_pixels,
            "coverage": round(coverage, 4),
        }
