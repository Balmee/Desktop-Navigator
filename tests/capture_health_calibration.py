
from __future__ import annotations

import argparse
import json
import time

from datetime import datetime
from pathlib import Path

from aura.perception.deadlock_observer import DeadlockObserver
from aura.perception.health_bar import HealthBarDetector


def main():

    parser = argparse.ArgumentParser(
        description="Capture Deadlock health calibration samples."
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    observer = DeadlockObserver()
    detector = HealthBarDetector()

    print("Switch to Deadlock.")
    print(f"Capturing in {args.delay} seconds...")

    time.sleep(args.delay)

    image = observer.capture_game_window()

    if image is None:
        print("Capture cancelled: Deadlock is not foregrounded.")
        return

    captured_at = datetime.now().isoformat()

    start = time.perf_counter()

    result = detector.segment(image)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    sample_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output = (
        Path("data/calibration/health")
        / sample_id
    )

    output.mkdir(parents=True, exist_ok=True)

    image.save(output / "source.png")

    result["crop"].save(
        output / "health_crop.png"
    )

    result["mask"].save(
        output / "mask.png"
    )

    result["overlay"].save(
        output / "overlay.png"
    )

    metadata = {
        "sample_id": sample_id,
        "captured_at": captured_at,
        "resolution": {
            "width": image.width,
            "height": image.height,
        },
        "ground_truth": {
            "current_health": None,
            "maximum_health": None,
            "health_fraction": None,
        },
        "segmentation": {
            "foreground_pixels": result["foreground_pixels"],
            "geometry_pixels": result["geometry_pixels"],
            "coverage": result["coverage"],
            "components": result["components"],
            "processing_ms": round(elapsed_ms, 3),
        },
    }

    print("\nCapture successful.")
    print(f"Saved to: {output.resolve()}")

    print("\nInspect health_crop.png before entering values.")
    print("Use the health shown in the captured image,")
    print("not the player's current live health.")

    try:

        current_health = int(
            input("\nCaptured current health: ")
        )

        maximum_health = int(
            input("Captured maximum health: ")
        )

        if maximum_health <= 0:
            raise ValueError("Maximum health must be positive.")

        if not 0 <= current_health <= maximum_health:
            raise ValueError("Invalid current health.")

        metadata["ground_truth"] = {
            "current_health": current_health,
            "maximum_health": maximum_health,
            "health_fraction": round(
                current_health / maximum_health,
                6,
            ),
        }

    except (ValueError, KeyboardInterrupt):

        print("\nSample saved without ground-truth labels.")

    finally:

        with open(
            output / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

    print("\n========== CALIBRATION SAMPLE ==========\n")

    print(json.dumps(metadata, indent=4))

    print("\n========================================")


if __name__ == "__main__":
    main()
