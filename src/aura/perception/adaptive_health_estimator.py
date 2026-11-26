
from __future__ import annotations

import time

import cv2
import numpy as np

from PIL import Image

from aura.perception.health_bar_tracker import (
    HealthBarTracker,
)


class AdaptiveHealthEstimator:
    """
    Estimates health using dynamically registered
    health bar geometry.
    """

    # Approximate interior polygon coordinates
    # in each original template's coordinate space.
    GEOMETRY = {
        "normal": {
            "size": (169, 270),
            "polygon": [
                (20, 46),
                (66, 17),
                (106, 262),
                (91, 264),
            ],
        },
        "critical": {
            "size": (116, 278),
            "polygon": [
                (20, 49),
                (71, 18),
                (108, 269),
                (94, 272),
            ],
        },
    }

    MIN_MATCH_SCORE = 0.70
    MIN_ROW_OCCUPANCY = 0.45
    MAX_ROW_GAP = 5

    MAX_ANCHOR_SEARCH_FRACTION = 0.08
    MIN_ANCHOR_SEARCH_PIXELS = 8

    def __init__(self):
        self.tracker = HealthBarTracker()

    def estimate(
        self,
        image: Image.Image,
        maximum_health: int | None = None,
    ) -> dict:

        start = time.perf_counter()

        if maximum_health is not None:
            if maximum_health <= 0:
                raise ValueError(
                    "Maximum health must be positive."
                )

        tracking = self.tracker.locate(image)

        candidate = tracking["best_candidate"]

        def unknown(reason):
            return {
                "detected": False,
                "health_fraction": None,
                "health_percentage": None,
                "estimated_current_health": None,
                "maximum_health": maximum_health,
                "quality": "unknown",
                "reason": reason,
                "tracking": candidate,
                "processing_ms": round(
                    (time.perf_counter() - start) * 1000,
                    3,
                ),
            }

        if not tracking["detected"]:
            return unknown("Health bar not located.")

        if candidate["score"] < self.MIN_MATCH_SCORE:
            return unknown("Template match too weak.")

        search_crop = self.tracker.localiser.crop(image)

        rgb = np.array(search_crop.convert("RGB"))

        name = candidate["template"]

        geometry = self.GEOMETRY[name]

        reference_width, reference_height = (
            geometry["size"]
        )

        scale_x = (
            candidate["width"] / reference_width
        )

        scale_y = (
            candidate["height"] / reference_height
        )

        x = candidate["x"]
        y = candidate["y"]

        polygon = np.array([
            [
                round(x + px * scale_x),
                round(y + py * scale_y),
            ]
            for px, py in geometry["polygon"]
        ], dtype=np.int32)

        geometry_mask = np.zeros(
            rgb.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            geometry_mask,
            [polygon],
            255,
        )

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        # Pale, relatively low-saturation health fill.
        pale_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 145]),
            np.array([179, 100, 255]),
        )

        fill_mask = cv2.bitwise_and(
            pale_mask,
            geometry_mask,
        )

        debug_overlay = rgb.copy()

        selected = fill_mask > 0

        debug_overlay[selected] = (
            0.4 * debug_overlay[selected]
            + 0.6 * np.array([0, 255, 0])
        ).astype(np.uint8)

        cv2.polylines(
            debug_overlay,
            [polygon],
            True,
            (0, 255, 255),
            2,
        )

        debug_images = {
            "geometry_mask": Image.fromarray(
                geometry_mask
            ),
            "fill_mask": Image.fromarray(
                fill_mask
            ),
            "overlay": Image.fromarray(
                debug_overlay
            ),
        }

        geometry_rows = np.count_nonzero(
            geometry_mask,
            axis=1,
        )

        fill_rows = np.count_nonzero(
            fill_mask,
            axis=1,
        )

        valid_rows = np.flatnonzero(
            geometry_rows >= 3
        )

        if len(valid_rows) < 20:
            return unknown("Invalid health bar geometry.")

        occupancy = np.divide(
            fill_rows,
            geometry_rows,
            out=np.zeros(
                len(fill_rows),
                dtype=np.float32,
            ),
            where=geometry_rows > 0,
        )

        occupied = (
            occupancy >= self.MIN_ROW_OCCUPANCY
        )

        top = int(valid_rows[0])
        bottom = int(valid_rows[-1])

        full_height = bottom - top + 1

        gap_limit = max(
            2,
            round(self.MAX_ROW_GAP * scale_y),
        )

        # Locate fill close to the bottom of the bar.
        
        # Search for valid fill near the predicted bottom.
        anchor_search = max(
            self.MIN_ANCHOR_SEARCH_PIXELS,
            round(
                full_height
                * self.MAX_ANCHOR_SEARCH_FRACTION
            ),
        )

        anchor_search = min(
            anchor_search,
            full_height,
        )

        anchor = None

        search_limit = max(
            top,
            bottom - anchor_search + 1,
        )

        for row in range(
            bottom,
            search_limit - 1,
            -1,
        ):
            if occupied[row]:
                anchor = row
                break

        if anchor is None:

            failure = unknown(
                "No bottom-anchored health fill detected."
            )

            failure.update(debug_images)

            failure["measurement"] = {
                "predicted_bottom": bottom,
                "predicted_top": top,
                "full_height": full_height,
                "anchor_search_pixels": anchor_search,
                "anchor_search_limit": search_limit,
            }

            return failure

        anchor_offset = bottom - anchor


        # Follow the fill upwards, tolerating
        # small gaps caused by HUD tick marks.
        for row in range(anchor, top - 1, -1):

            if occupied[row]:
                highest = row
                gap = 0
            else:
                gap += 1

                if gap > gap_limit:
                    break

        
        # Use the observed bottom of the fill rather
        # than counting the unoccupied registration gap.
        effective_bottom = anchor

        effective_full_height = (
            effective_bottom - top + 1
        )

        if effective_full_height <= 0:

            failure = unknown(
                "Invalid corrected health geometry."
            )

            failure.update(debug_images)

            return failure

        fill_height = (
            effective_bottom - highest + 1
        )

        fraction = float(
            np.clip(
                fill_height / effective_full_height,
                0.0,
                1.0,
            )
        )


        estimated_health = (
            round(maximum_health * fraction)
            if maximum_health is not None
            else None
        )

        overlay = rgb.copy()

        selected = fill_mask > 0

        overlay[selected] = (
            0.4 * overlay[selected]
            + 0.6 * np.array([0, 255, 0])
        ).astype(np.uint8)

        cv2.polylines(
            overlay,
            [polygon],
            True,
            (0, 255, 255),
            2,
        )

        cv2.line(
            overlay,
            (int(polygon[:, 0].min()), highest),
            (int(polygon[:, 0].max()), highest),
            (255, 0, 255),
            1,
        )

        return {
            "detected": True,
            "health_fraction": round(fraction, 6),
            "health_percentage": round(
                fraction * 100,
                2,
            ),
            "estimated_current_health": estimated_health,
            "maximum_health": maximum_health,
            "quality": "provisional",
            "tracking": candidate,
            "measurement": {
                "fill_height": fill_height,
                "full_height": effective_full_height,
                "original_full_height": full_height,
                "highest_fill_row": highest,
                "bottom_row": effective_bottom,
                "predicted_bottom_row": bottom,
                "anchor_offset": anchor_offset,
                "anchor_search_pixels": anchor_search,
            },
            "method": "adaptive_template_geometry",
            "processing_ms": round(
                (time.perf_counter() - start) * 1000,
                3,
            ),
            "geometry_mask": Image.fromarray(
                geometry_mask
            ),
            "fill_mask": Image.fromarray(fill_mask),
            "overlay": Image.fromarray(overlay),
        }
