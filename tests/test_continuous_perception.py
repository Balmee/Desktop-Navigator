
from __future__ import annotations

import argparse
import json
import time

from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from aura.perception.continuous_perception import (
    ContinuousPerceptionEngine,
    PerceptionObservation,
)


OUTPUT_ROOT = Path(
    "data/validation/continuous_perception"
)


def percentile(values, percentage):

    if not values:
        return None

    ordered = sorted(values)

    index = (
        len(ordered) - 1
    ) * percentage / 100

    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = index - lower

    return round(
        ordered[lower]
        + (
            ordered[upper] - ordered[lower]
        ) * fraction,
        3,
    )


def average(values):

    if not values:
        return None

    return round(mean(values), 3)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    if args.duration <= 0 or args.hz <= 0:
        parser.error(
            "Duration and frequency must be positive."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output = OUTPUT_ROOT / session_id

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = ContinuousPerceptionEngine(
        target_hz=args.hz,
    )

    observations = []

    last_print = 0.0
    previous_status = None

    def on_observation(
        observation: PerceptionObservation,
    ):

        nonlocal last_print
        nonlocal previous_status

        observations.append(
            observation.to_dict()
        )

        now = time.monotonic()

        status_changed = (
            observation.status
            != previous_status
        )

        should_print = (
            now - last_print >= 1.0
            or status_changed
        )

        previous_status = observation.status

        if not should_print:
            return

        last_print = now

        if observation.health_detected:

            health = (
                f"{observation.health_percentage:.2f}%"
            )

        else:
            health = "UNKNOWN"

        print(
            f"Frame {observation.frame_id:04d}"
            f" | {observation.status:18s}"
            f" | Health: {health:8s}"
            f" | Capture: "
            f"{observation.capture_ms:6.2f} ms"
            f" | Perception: "
            f"{observation.perception_ms:6.2f} ms"
            f" | Total: "
            f"{observation.total_processing_ms:6.2f} ms",
            flush=True,
        )

    print(
        "\n========== CONTINUOUS PERCEPTION TEST ==========\n"
    )

    print(f"Session: {session_id}")
    print(f"Target frequency: {args.hz} Hz")
    print(f"Duration: {args.duration} seconds")

    print(
        f"\nSwitch to Deadlock within "
        f"{args.delay:.1f} seconds...\n"
    )

    time.sleep(args.delay)

    start = time.monotonic()

    try:

        engine.run(
            on_observation=on_observation,
            duration=args.duration,
        )

    except KeyboardInterrupt:

        print("\nObservation interrupted.")

        engine.stop()

    elapsed = time.monotonic() - start

    counts = Counter(
        item["status"]
        for item in observations
    )

    captured = [
        item for item in observations
        if item["frame_available"]
    ]

    valid = [
        item for item in observations
        if item["health_detected"]
    ]

    capture_times = [
        item["capture_ms"]
        for item in captured
    ]

    perception_times = [
        item["perception_ms"]
        for item in captured
    ]

    processing_times = [
        item["total_processing_ms"]
        for item in observations
    ]

    captured_processing_times = [
        item["total_processing_ms"]
        for item in captured
    ]

    budget_ms = 1000 / args.hz

    over_budget = sum(
        value > budget_ms
        for value in processing_times
    )

    health_values = [
        item["health_percentage"]
        for item in valid
    ]

    summary = {
        "session_id": session_id,
        "target_hz": args.hz,
        "actual_duration_seconds": round(
            elapsed, 3
        ),
        "total_observations": len(observations),
        "captured_frames": len(captured),
        "valid_health_observations": len(valid),
        "effective_observation_hz": round(
            len(observations) / elapsed,
            3,
        ),
        "effective_capture_hz": round(
            len(captured) / elapsed,
            3,
        ),
        "status_counts": dict(counts),
        "health_detection_rate": (
            round(
                len(valid) / len(captured) * 100,
                2,
            )
            if captured else None
        ),
        "mean_capture_ms": average(
            capture_times
        ),
        "mean_perception_ms": average(
            perception_times
        ),
        "mean_total_processing_ms": average(
            processing_times
        ),
        "mean_captured_processing_ms": average(
            captured_processing_times
        ),
        "p95_captured_processing_ms": percentile(
            captured_processing_times,
            95,
        ),
        "max_total_processing_ms": (
            round(max(processing_times), 3)
            if processing_times else None
        ),
        "frame_budget_ms": round(
            budget_ms, 3
        ),
        "over_budget_observations": over_budget,
        "over_budget_rate": (
            round(
                over_budget
                / len(observations) * 100,
                2,
            )
            if observations else None
        ),
        "minimum_health_percentage": (
            min(health_values)
            if health_values else None
        ),
        "maximum_health_percentage": (
            max(health_values)
            if health_values else None
        ),
    }

    report = {
        "summary": summary,
        "observations": observations,
    }

    report_path = output / "report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(
        "\n========== PERCEPTION SUMMARY ==========\n"
    )

    print(json.dumps(summary, indent=4))

    print(
        "\n========================================"
    )

    print(f"\nReport: {report_path.resolve()}")


if __name__ == "__main__":
    main()
