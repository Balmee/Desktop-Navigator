
from __future__ import annotations

from PIL import Image

from aura.perception.health_bar import HealthBarDetector


class HealthEstimator:
    """
    Estimates the player's remaining health fraction
    using calibrated health bar geometry.
    """

    REFERENCE_CROP_HEIGHT = 291
    REFERENCE_FULL_HEIGHT = 257
    REFERENCE_BOTTOM = 278

    MIN_COMPONENT_AREA = 30
    BOTTOM_TOLERANCE = 10

    def __init__(self):
        self.detector = HealthBarDetector()

    def estimate(
        self,
        image: Image.Image,
        maximum_health: int | None = None,
    ) -> dict:

        segmentation = self.detector.segment(image)

        return self.estimate_from_segmentation(
            segmentation,
            maximum_health=maximum_health,
        )

    def estimate_from_segmentation(
        self,
        segmentation: dict,
        maximum_health: int | None = None,
    ) -> dict:

        if maximum_health is not None and maximum_health <= 0:
            raise ValueError("Maximum health must be positive.")

        crop_height = segmentation["crop"].height

        scale = (
            crop_height / self.REFERENCE_CROP_HEIGHT
        )

        full_height = (
            self.REFERENCE_FULL_HEIGHT * scale
        )

        expected_bottom = (
            self.REFERENCE_BOTTOM * scale
        )

        bottom_tolerance = (
            self.BOTTOM_TOLERANCE * scale
        )

        candidates = []

        for component in segmentation["components"]:

            if component["area"] < (
                self.MIN_COMPONENT_AREA * scale ** 2
            ):
                continue

            bottom = (
                component["y"] + component["height"]
            )

            # Reject components that do not extend
            # to the expected bottom of the health bar.
            if abs(bottom - expected_bottom) > bottom_tolerance:
                continue

            candidates.append(component)

        if not candidates:

            return {
                "detected": False,
                "health_fraction": None,
                "health_percentage": None,
                "estimated_current_health": None,
                "maximum_health": maximum_health,
                "quality": "unknown",
                "reason": (
                    "No valid bottom-anchored health "
                    "bar component detected."
                ),
            }

        component = max(
            candidates,
            key=lambda item: item["area"],
        )

        detected_height = component["height"]

        raw_fraction = detected_height / full_height

        if raw_fraction > 1.05:

            return {
                "detected": False,
                "health_fraction": None,
                "health_percentage": None,
                "estimated_current_health": None,
                "maximum_health": maximum_health,
                "quality": "unknown",
                "reason": (
                    "Detected component exceeds the "
                    "calibrated health bar height."
                ),
            }

        fraction = max(
            0.0,
            min(1.0, raw_fraction),
        )

        estimated_health = None

        if maximum_health is not None:

            estimated_health = round(
                maximum_health * fraction
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
            "measurement": {
                "component_height": detected_height,
                "reference_full_height": round(
                    full_height,
                    2,
                ),
                "component_area": component["area"],
                "coverage": segmentation["coverage"],
            },
            "method": "calibrated_component_height",
        }
