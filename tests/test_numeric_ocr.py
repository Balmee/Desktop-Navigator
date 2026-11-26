
import json
import time

from pathlib import Path
from PIL import Image

from aura.perception.numeric_ocr import NumericOCR


REGIONS = {
    "current_health": (0.245, 0.665, 0.285, 0.715),
    "maximum_health": (0.282, 0.690, 0.310, 0.727),
    "player_hp_stat": (0.107, 0.747, 0.143, 0.785),
}


def main():

    source = Path("data/debug/hud/source_frame.png")

    if not source.exists():
        raise FileNotFoundError(source)

    image = Image.open(source).convert("RGB")

    ocr = NumericOCR()

    output_directory = Path("data/debug/numeric")
    output_directory.mkdir(parents=True, exist_ok=True)

    results = {}

    start_time = time.perf_counter()

    for name, region in REGIONS.items():

        left, top, right, bottom = region

        crop = image.crop((
            int(left * image.width),
            int(top * image.height),
            int(right * image.width),
            int(bottom * image.height),
        ))

        crop.save(output_directory / f"{name}.png")

        variants = ocr.preprocess(crop)

        for method, processed in variants.items():

            Image.fromarray(processed).save(
                output_directory / f"{name}_{method}.png"
            )

        results[name] = ocr.extract(crop)

    elapsed = time.perf_counter() - start_time

    print("\n========== NUMERIC OCR ==========\n")

    print(json.dumps(results, indent=4))

    print("\n=================================")

    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Debug directory: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
