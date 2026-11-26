
import json
import time
from pathlib import Path

from aura.perception.deadlock_observer import DeadlockObserver
from aura.perception.deadlock_hud import DeadlockHUD


def main():

    observer = DeadlockObserver()
    hud = DeadlockHUD()

    print("Capturing Deadlock frame...")

    image = observer.capture_game_window()

    if image is None:
        print("Deadlock is not foregrounded.")
        return

    print(f"Resolution: {image.width}x{image.height}")

    debug_directory = Path("data/debug/hud")
    debug_directory.mkdir(parents=True, exist_ok=True)

    # Save the exact frame used for this diagnostic test.
    image.save(debug_directory / "source_frame.png")

    print("\nExtracting HUD regions...")

    start_time = time.perf_counter()

    results = hud.analyse(image)

    elapsed = time.perf_counter() - start_time

    # Save diagnostic crops for visual inspection.
    for region_name in hud.REGIONS:

        cropped = hud.crop_region(image, region_name)
        processed = hud.preprocess(cropped)

        cropped.save(
            debug_directory / f"{region_name}_original.png"
        )

        processed.save(
            debug_directory / f"{region_name}_processed.png"
        )

    print("\n========== DEADLOCK HUD ANALYSIS ==========\n")

    print(json.dumps(
        results,
        indent=4,
        ensure_ascii=False,
    ))

    print("\n===========================================")

    print(f"\nOCR processing time: {elapsed:.3f} seconds")
    print(f"Debug images: {debug_directory.resolve()}")


if __name__ == "__main__":
    main()
