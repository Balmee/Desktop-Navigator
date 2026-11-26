
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from aura.perception.health_bar_localiser import (
    HealthBarLocaliser,
)


REFERENCES = {
    "normal": {
        "sample": "20261126_171500_438918",
        "bounds": (130, 144, 299, 414),
    },
    "critical": {
        "sample": "20261126_173124_701095",
        "bounds": (127, 139, 243, 417),
    },
}


def main():

    dataset = Path("data/calibration/health")

    output = Path("data/templates/health")
    output.mkdir(parents=True, exist_ok=True)

    localiser = HealthBarLocaliser()

    for name, reference in REFERENCES.items():

        source = (
            dataset
            / reference["sample"]
            / "source.png"
        )

        with Image.open(source) as frame:
            image = frame.convert("RGB")

        result = localiser.locate(image)

        original = result["crop"].crop(
            reference["bounds"]
        )

        mask_name = (
            "green_mask"
            if name == "normal"
            else "critical_mask"
        )

        colour_mask = result[mask_name].crop(
            reference["bounds"]
        )

        binary = np.array(colour_mask)

        # Remove isolated noise.
        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        if count <= 1:
            raise RuntimeError(
                f"No template component found: {name}"
            )

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        template = np.zeros_like(binary)
        template[labels == largest] = 255

        original.save(
            output / f"{name}_original.png"
        )

        Image.fromarray(template).save(
            output / f"{name}_template.png"
        )

        print(
            f"Created {name} template: "
            f"{template.shape[1]}x{template.shape[0]}"
        )

    print(f"\nTemplates saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
