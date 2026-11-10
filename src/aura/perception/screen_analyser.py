from PIL import Image, ImageChops, ImageStat


class ScreenChangeDetector:

    def __init__(self, threshold: float = 0.035):

        self.threshold = threshold
        self.previous_image = None

    def prepare_image(self, image: Image.Image):

        return image.convert("L").resize(
            (320, 180),
            Image.Resampling.LANCZOS
        )

    def compare(self, current_image: Image.Image) -> dict:

        current = self.prepare_image(current_image)

        if self.previous_image is None:

            self.previous_image = current

            return {
                "changed": True,
                "difference": 1.0,
                "reason": "initial_capture"
            }

        difference_image = ImageChops.difference(
            self.previous_image,
            current
        )

        difference = ImageStat.Stat(
            difference_image
        ).mean[0] / 255.0

        changed = difference >= self.threshold

        self.previous_image = current

        return {
            "changed": changed,
            "difference": round(difference, 4),
            "reason": "visual_change" if changed else "no_significant_change"
        }

    def reset(self):

        self.previous_image = None