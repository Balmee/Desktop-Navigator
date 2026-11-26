
import json
import time

from pathlib import Path
from statistics import mean

from PIL import Image

from aura.perception.health_bar_tracker import (
    HealthBarTracker,
)


def main():

    dataset = Path("data/calibration/health")

    output = Path("data/debug/health_tracking")
    output.mkdir(parents=True, exist_ok=True)

    tracker = HealthBarTracker()

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

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        with Image.open(source) as frame:
            image = frame.convert("RGB")

        start = time.perf_counter()

        result = tracker.locate(image)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        sample_output = output / directory.name

        sample_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        result["overlay"].save(
            sample_output / "overlay.png"
        )

        report = {
            "sample": directory.name,
            "ground_truth": metadata["ground_truth"],
            "detected": result["detected"],
            "best_candidate": result["best_candidate"],
            "alternatives": result["candidates"],
            "processing_ms": round(elapsed, 3),
        }

        reports.append(report)

        print(json.dumps(report, indent=4))

    summary = {
        "samples": len(reports),
        "accepted_matches": sum(
            report["detected"]
            for report in reports
        ),
        "mean_processing_ms": (
            round(
                mean(
                    report["processing_ms"]
                    for report in reports
                ),
                3,
            )
            if reports else None
        ),
    }

    final_report = {
        "samples": reports,
        "summary": summary,
    }

    (output / "report.json").write_text(
        json.dumps(final_report, indent=4),
        encoding="utf-8",
    )

    print("\n========== TRACKING SUMMARY ==========\n")
    print(json.dumps(summary, indent=4))
    print(f"\nResults: {output.resolve()}")


if __name__ == "__main__":
    main()
