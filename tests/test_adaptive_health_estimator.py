
import json

from pathlib import Path
from statistics import mean

from PIL import Image

from aura.perception.health_estimator import (
    HealthEstimator,
)

from aura.perception.adaptive_health_estimator import (
    AdaptiveHealthEstimator,
)


def main():

    dataset = Path("data/calibration/health")

    output = Path(
        "data/debug/adaptive_health"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = HealthEstimator()
    adaptive = AdaptiveHealthEstimator()

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
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        truth = metadata["ground_truth"]

        current = truth["current_health"]
        maximum = truth["maximum_health"]

        if not maximum or current is None:
            continue

        actual = current / maximum * 100

        with Image.open(source) as frame:
            image = frame.convert("RGB")

        old = baseline.estimate(
            image,
            maximum_health=maximum,
        )

        new = adaptive.estimate(
            image,
            maximum_health=maximum,
        )

        sample_output = output / directory.name

        sample_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name in (
            "geometry_mask",
            "fill_mask",
            "overlay",
        ):
            if name in new:
                new[name].save(
                    sample_output / f"{name}.png"
                )

        old_percentage = old.get(
            "health_percentage"
        )

        new_percentage = new.get(
            "health_percentage"
        )

        report = {
            "sample": directory.name,
            "actual_health": current,
            "maximum_health": maximum,
            "actual_percentage": round(actual, 3),
            "baseline_percentage": old_percentage,
            "adaptive_percentage": new_percentage,
            "baseline_error_pp": (
                round(
                    abs(actual - old_percentage),
                    3,
                )
                if old_percentage is not None
                else None
            ),
            "adaptive_error_pp": (
                round(
                    abs(actual - new_percentage),
                    3,
                )
                if new_percentage is not None
                else None
            ),
            "adaptive_detected": new["detected"],
            "tracking": new.get("tracking"),
            "measurement": new.get("measurement"),
            "reason": new.get("reason"),
            "processing_ms": new["processing_ms"],
        }

        reports.append(report)

        print(json.dumps(report, indent=4))

    old_errors = [
        item["baseline_error_pp"]
        for item in reports
        if item["baseline_error_pp"] is not None
    ]

    new_errors = [
        item["adaptive_error_pp"]
        for item in reports
        if item["adaptive_error_pp"] is not None
    ]

    summary = {
        "samples": len(reports),
        "baseline_detections": len(old_errors),
        "adaptive_detections": len(new_errors),
        "baseline_mae_pp": (
            round(mean(old_errors), 3)
            if old_errors else None
        ),
        "adaptive_mae_pp": (
            round(mean(new_errors), 3)
            if new_errors else None
        ),
        "baseline_max_error_pp": (
            max(old_errors)
            if old_errors else None
        ),
        "adaptive_max_error_pp": (
            max(new_errors)
            if new_errors else None
        ),
        "adaptive_mean_processing_ms": (
            round(mean([
                item["processing_ms"]
                for item in reports
            ]), 3)
            if reports else None
        ),
    }

    result = {
        "samples": reports,
        "summary": summary,
    }

    (output / "report.json").write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    print("\n========== ADAPTIVE HEALTH SUMMARY ==========\n")

    print(json.dumps(summary, indent=4))

    print(f"\nResults: {output.resolve()}")


if __name__ == "__main__":
    main()
