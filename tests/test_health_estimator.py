
from __future__ import annotations

import json
import time

from pathlib import Path
from statistics import mean

from PIL import Image

from aura.perception.health_estimator import HealthEstimator


def main():

    dataset = Path("data/calibration/health")

    if not dataset.exists():
        raise FileNotFoundError(dataset)

    estimator = HealthEstimator()

    results = []

    percentage_errors = []
    health_errors = []
    processing_times = []

    total_samples = 0
    failed_samples = 0

    for directory in sorted(dataset.iterdir()):

        if not directory.is_dir():
            continue

        metadata_path = directory / "metadata.json"
        source_path = directory / "source.png"

        if not metadata_path.exists():
            continue

        if not source_path.exists():
            continue

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        ground_truth = metadata["ground_truth"]

        current_health = ground_truth["current_health"]
        maximum_health = ground_truth["maximum_health"]

        if current_health is None:
            continue

        if maximum_health is None or maximum_health <= 0:
            continue

        if not 0 <= current_health <= maximum_health:
            continue

        total_samples += 1

        with Image.open(source_path) as source:
            image = source.convert("RGB")

        start = time.perf_counter()

        prediction = estimator.estimate(
            image,
            maximum_health=maximum_health,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        processing_times.append(elapsed_ms)

        if not prediction["detected"]:

            failed_samples += 1

            results.append({
                "sample": directory.name,
                "detected": False,
                "reason": prediction.get("reason"),
                "processing_ms": round(elapsed_ms, 3),
            })

            continue

        actual_percentage = (
            current_health / maximum_health
        ) * 100

        predicted_percentage = (
            prediction["health_percentage"]
        )

        percentage_error = abs(
            actual_percentage - predicted_percentage
        )

        health_error = abs(
            current_health
            - prediction["estimated_current_health"]
        )

        percentage_errors.append(percentage_error)
        health_errors.append(health_error)

        results.append({
            "sample": directory.name,
            "detected": True,
            "actual_health": current_health,
            "maximum_health": maximum_health,
            "estimated_health": (
                prediction["estimated_current_health"]
            ),
            "actual_percentage": round(
                actual_percentage,
                2,
            ),
            "estimated_percentage": predicted_percentage,
            "absolute_error_pp": round(
                percentage_error,
                3,
            ),
            "absolute_error_hp": health_error,
            "processing_ms": round(elapsed_ms, 3),
        })

    summary = {
        "total_samples": total_samples,
        "successful_detections": len(percentage_errors),
        "failed_detections": failed_samples,
        "detection_rate": (
            round(
                len(percentage_errors) / total_samples * 100,
                2,
            )
            if total_samples else None
        ),
        "mean_absolute_error_pp": (
            round(mean(percentage_errors), 3)
            if percentage_errors else None
        ),
        "maximum_absolute_error_pp": (
            round(max(percentage_errors), 3)
            if percentage_errors else None
        ),
        "mean_absolute_error_hp": (
            round(mean(health_errors), 3)
            if health_errors else None
        ),
        "mean_processing_ms": (
            round(mean(processing_times), 3)
            if processing_times else None
        ),
    }

    report = {
        "samples": results,
        "summary": summary,
    }

    print("\n========== HEALTH ESTIMATOR EVALUATION ==========\n")

    print(json.dumps(report, indent=4))

    output = Path(
        "data/calibration/health_evaluation.json"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(report, file, indent=4)

    print("\n=================================================")

    print(f"Evaluation saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
