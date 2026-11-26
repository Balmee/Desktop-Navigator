
from __future__ import annotations

import json
import time

from pathlib import Path

from PIL import Image

from aura.perception.health_bar_localiser import (
    HealthBarLocaliser,
)


def main():

    dataset = Path("data/calibration/health")

    output_root = Path(
        "data/debug/health_localisation"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    localiser = HealthBarLocaliser()

    reports = []

    for directory in sorted(dataset.iterdir()):

        if not directory.is_dir():
            continue

        source = directory / "source.png"
        metadata_path = directory / "metadata.json"

        if not source.exists():
            continue

        if not metadata_path.exists():
            continue

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        with Image.open(source) as frame:
            image = frame.convert("RGB")

        start = time.perf_counter()

        result = localiser.locate(image)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        output = output_root / directory.name

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name in (
            "crop",
            "green_mask",
            "critical_mask",
            "overlay",
        ):

            result[name].save(
                output / f"{name}.png"
            )

        report = {
            "sample": directory.name,
            "ground_truth": metadata["ground_truth"],
            "candidate_count": len(
                result["candidates"]
            ),
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "hull"
                }
                for candidate in result["candidates"]
            ],
            "processing_ms": round(
                elapsed_ms,
                3,
            ),
        }

        reports.append(report)

        print(json.dumps(report, indent=4))

    with (
        output_root / "report.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(reports, file, indent=4)

    print("\n========== LOCALISATION COMPLETE ==========")

    print(f"Samples processed: {len(reports)}")

    print(f"Results: {output_root.resolve()}")


if __name__ == "__main__":
    main()
