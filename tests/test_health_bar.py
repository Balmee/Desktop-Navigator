
import json
import time

from pathlib import Path
from PIL import Image

from aura.perception.health_bar import HealthBarDetector


def main():

    source = Path(
        "data/debug/hud/source_frame.png"
    )

    image = Image.open(source).convert("RGB")

    detector = HealthBarDetector()

    start = time.perf_counter()

    result = detector.segment(image)

    elapsed = time.perf_counter() - start

    output = Path("data/debug/health_bar")
    output.mkdir(parents=True, exist_ok=True)

    result["crop"].save(
        output / "original.png"
    )

    result["mask"].save(
        output / "mask.png"
    )

    result["overlay"].save(
        output / "overlay.png"
    )

    result["geometry_mask"].save(
        output / "geometry_mask.png"
    )

    result["colour_mask"].save(
        output / "colour_mask.png"
    )

    print("\n========== HEALTH BAR SEGMENTATION ==========\n")

    print(json.dumps({
        "foreground_pixels": result["foreground_pixels"],
        "geometry_pixels": result["geometry_pixels"],
        "coverage": result["coverage"],
        "components": result["components"],
        "processing_ms": round(elapsed * 1000, 3),
    }, indent=4))

    print("\n=============================================")

    print(f"Debug images: {output.resolve()}")


if __name__ == "__main__":
    main()
