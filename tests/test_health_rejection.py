
from __future__ import annotations

import argparse
import json
import time

from datetime import datetime
from pathlib import Path

from aura.perception.deadlock_observer import (
    DeadlockObserver,
)

from aura.perception.adaptive_health_estimator import (
    AdaptiveHealthEstimator,
)


OUTPUT_ROOT = Path(
    "data/validation/health_rejection"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        required=True,
        help="Description of the current game state.",
    )

    parser.add_argument(
        "--expected",
        required=True,
        choices=["present", "absent"],
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    observer = DeadlockObserver()
    estimator = AdaptiveHealthEstimator()

    print("\n========== HEALTH HUD VALIDATION ==========")

    print(f"Scenario: {args.label}")
    print(f"Expected HUD: {args.expected}")

    print(
        f"\nSwitch to Deadlock within "
        f"{args.delay:.1f} seconds..."
    )

    time.sleep(args.delay)

    image = observer.capture_game_window()

    if image is None:

        print(
            "Capture skipped: Deadlock is not "
            "available for observation."
        )

        return

    print(
        f"Captured frame: "
        f"{image.width}x{image.height}"
    )

    result = estimator.estimate(image)

    detected = bool(
        result.get("detected", False)
    )

    expected_detection = (
        args.expected == "present"
    )

    passed = (
        detected == expected_detection
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    sample_directory = (
        OUTPUT_ROOT / timestamp
    )

    sample_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        sample_directory / "source.png"
    )

    for name in (
        "geometry_mask",
        "fill_mask",
        "overlay",
    ):

        if name in result:

            result[name].save(
                sample_directory / f"{name}.png"
            )

    report = {
        "sample": timestamp,
        "label": args.label,
        "expected_hud": args.expected,
        "detected": detected,
        "passed": passed,
        "health_percentage": result.get(
            "health_percentage"
        ),
        "quality": result.get("quality"),
        "reason": result.get("reason"),
        "tracking": result.get("tracking"),
        "measurement": result.get("measurement"),
        "processing_ms": result.get(
            "processing_ms"
        ),
    }

    metadata_path = (
        sample_directory / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    report_path = OUTPUT_ROOT / "report.json"

    if report_path.exists():

        existing = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        reports = existing.get("samples", [])

    else:
        reports = []

    reports.append(report)

    negatives = [
        item for item in reports
        if item["expected_hud"] == "absent"
    ]

    positives = [
        item for item in reports
        if item["expected_hud"] == "present"
    ]

    false_positives = sum(
        item["detected"]
        for item in negatives
    )

    false_negatives = sum(
        not item["detected"]
        for item in positives
    )

    summary = {
        "total_samples": len(reports),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "passed": sum(
            item["passed"]
            for item in reports
        ),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_rate": (
            round(
                false_positives
                / len(negatives) * 100,
                2,
            )
            if negatives else None
        ),
        "false_negative_rate": (
            round(
                false_negatives
                / len(positives) * 100,
                2,
            )
            if positives else None
        ),
    }

    report_path.write_text(
        json.dumps({
            "samples": reports,
            "summary": summary,
        }, indent=4),
        encoding="utf-8",
    )

    print("\n========== SAMPLE RESULT ==========\n")

    print(json.dumps(report, indent=4))

    print("\n========== VALIDATION SUMMARY ==========\n")

    print(json.dumps(summary, indent=4))

    print(
        "\nResult:",
        "PASS" if passed else "FAIL",
    )

    print(
        f"\nReport: {report_path.resolve()}"
    )


if __name__ == "__main__":
    main()
