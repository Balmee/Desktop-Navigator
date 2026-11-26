
from __future__ import annotations

import argparse
import json
import math
import time

from collections import Counter
from pathlib import Path
from statistics import pstdev

from aura.gameplay.state_builder import (
    GameplayStateBuilder,
)

from aura.gameplay.temporal_tracker import (
    TemporalStateTracker,
)

from aura.gameplay.signal_filter import (
    HealthSignalFilter,
)

from aura.perception.continuous_perception import (
    PerceptionObservation,
)


class TestPipeline:

    def __init__(self):

        self.builder = GameplayStateBuilder()
        self.tracker = TemporalStateTracker()
        self.filter = HealthSignalFilter()

        self.frame_id = 0

    def update(
        self,
        timestamp: float,
        health: float | None,
        status: str = "observed",
    ):

        self.frame_id += 1

        detected = (
            health is not None
            and status == "observed"
        )

        observation = PerceptionObservation(
            frame_id=self.frame_id,
            timestamp=1000.0 + timestamp,
            monotonic_timestamp=timestamp,
            status=status,
            frame_available=(
                status != "unavailable"
            ),
            health_detected=detected,
            health_fraction=(
                health / 100
                if detected else None
            ),
            health_percentage=(
                health if detected else None
            ),
            measurement_quality=(
                "provisional"
                if detected else "unknown"
            ),
            capture_ms=28.0,
            perception_ms=38.0,
            total_processing_ms=66.0,
        )

        state = self.builder.build(observation)

        temporal = self.tracker.update(state)

        return self.filter.update(temporal)


def test_initialisation():

    pipeline = TestPipeline()

    result = pipeline.update(0.0, 75.0)

    assert result.available
    assert result.raw_percentage == 75.0
    assert result.filtered_percentage == 75.0
    assert result.mode == "initialised"

    print("PASS: Signal initialisation")


def test_deadband():

    pipeline = TestPipeline()

    pipeline.update(0.0, 75.0)

    result = pipeline.update(0.1, 75.3)

    assert result.mode == "deadband"
    assert result.filtered_percentage == 75.0
    assert result.raw_percentage == 75.3

    print("PASS: Small fluctuation suppression")


def test_noise_reduction():

    pipeline = TestPipeline()

    raw_values = []
    filtered_values = []

    for i in range(101):

        raw = (
            70.0
            + 0.8 * math.sin(i * 2.3)
        )

        result = pipeline.update(
            i / 10,
            raw,
        )

        raw_values.append(
            result.raw_percentage
        )

        filtered_values.append(
            result.filtered_percentage
        )

    raw_noise = pstdev(raw_values[10:])

    filtered_noise = pstdev(
        filtered_values[10:]
    )

    assert filtered_noise < raw_noise

    reduction = (
        1 - filtered_noise / raw_noise
    ) * 100

    print(
        "PASS: Noise reduction "
        f"({reduction:.2f}% reduction)"
    )


def test_abrupt_health_loss():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    result = pipeline.update(0.1, 30.0)

    assert result.abrupt_change
    assert not result.abrupt_change_confirmed

    assert result.filtered_percentage == 30.0

    assert result.filtered_change_pp == -50.0

    assert result.mode == "abrupt"

    print("PASS: Immediate abrupt health loss")


def test_abrupt_confirmation():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(0.1, 30.0)

    result = pipeline.update(0.2, 30.5)

    assert result.abrupt_change_confirmed

    print("PASS: Abrupt change corroboration")


def test_transient_spike():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    spike = pipeline.update(0.1, 30.0)

    recovery = pipeline.update(0.2, 80.0)

    assert spike.abrupt_change
    assert not spike.abrupt_change_confirmed

    assert not recovery.abrupt_change_confirmed

    assert recovery.filtered_percentage == 80.0

    print("PASS: Transient spike recovery")


def test_abrupt_healing():

    pipeline = TestPipeline()

    pipeline.update(0.0, 20.0)

    result = pipeline.update(0.1, 65.0)

    assert result.abrupt_change
    assert result.filtered_percentage == 65.0

    print("PASS: Immediate abrupt health increase")


def test_gradual_change():

    pipeline = TestPipeline()

    for i in range(51):

        raw = 30.0 + i * 0.4

        result = pipeline.update(
            i / 10,
            raw,
        )

    assert result.filtered_percentage > 48.0

    assert abs(
        result.filtered_percentage
        - result.raw_percentage
    ) < 2.0

    print("PASS: Gradual change tracking")


def test_missing_measurement():

    pipeline = TestPipeline()

    pipeline.update(0.0, 75.0)

    result = pipeline.update(
        0.1,
        None,
        "hud_unknown",
    )

    assert not result.available

    assert result.raw_percentage is None
    assert result.filtered_percentage is None

    print("PASS: Missing measurements remain unknown")


def test_long_gap_reset():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(
        1.0,
        None,
        "hud_unknown",
    )

    result = pipeline.update(1.1, 30.0)

    assert result.mode == "initialised"

    assert result.filtered_percentage == 30.0

    assert result.filtered_change_pp is None

    print("PASS: Long measurement gap reset")


def test_foreground_loss():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    result = pipeline.update(
        0.1,
        None,
        "unavailable",
    )

    assert not result.available

    result = pipeline.update(0.2, 25.0)

    assert result.mode == "initialised"
    assert result.filtered_percentage == 25.0

    assert not result.abrupt_change

    print("PASS: Foreground loss reset")


def test_health_endpoints():

    pipeline = TestPipeline()

    pipeline.update(0.0, 0.3)

    result = pipeline.update(0.1, 0.0)

    assert result.filtered_percentage == 0.0

    pipeline = TestPipeline()

    pipeline.update(0.0, 99.7)

    result = pipeline.update(0.1, 100.0)

    assert result.filtered_percentage == 100.0

    print("PASS: Exact health endpoints")


def test_noise_estimation():

    pipeline = TestPipeline()

    for i in range(30):

        result = pipeline.update(
            i / 10,
            70.0 + 0.4 * math.sin(i),
        )

    assert result.estimated_noise_pp is not None

    assert result.estimated_noise_pp >= 0.0

    print("PASS: Robust noise estimation")


def run_synthetic_tests():

    print(
        "\n========== SYNTHETIC TESTS ==========\n"
    )

    tests = [
        test_initialisation,
        test_deadband,
        test_noise_reduction,
        test_abrupt_health_loss,
        test_abrupt_confirmation,
        test_transient_spike,
        test_abrupt_healing,
        test_gradual_change,
        test_missing_measurement,
        test_long_gap_reset,
        test_foreground_loss,
        test_health_endpoints,
        test_noise_estimation,
    ]

    for test in tests:

        test()

    print(
        f"\nAll {len(tests)} synthetic tests passed."
    )


def replay_report(path: Path):

    report = json.loads(
        path.read_text(encoding="utf-8")
    )

    builder = GameplayStateBuilder()
    tracker = TemporalStateTracker()
    signal_filter = HealthSignalFilter()

    modes = Counter()

    processing_times = []
    residuals = []

    signals = []

    abrupt_count = 0
    confirmed_count = 0
    valid_count = 0

    for raw in report["observations"]:

        observation = PerceptionObservation(
            **raw
        )

        state = builder.build(observation)

        temporal = tracker.update(state)

        start = time.perf_counter()

        signal = signal_filter.update(
            temporal
        )

        processing_times.append(
            (time.perf_counter() - start)
            * 1000
        )

        modes[signal.mode] += 1

        if signal.available:

            valid_count += 1

            assert state.health_available

            assert (
                signal.raw_percentage
                == state.health_percentage
            )

            assert (
                0.0
                <= signal.filtered_percentage
                <= 100.0
            )

            residuals.append(
                abs(signal.residual_pp)
            )

        else:

            assert not state.health_available

            assert (
                signal.filtered_percentage
                is None
            )

        if signal.abrupt_change:

            abrupt_count += 1

            assert (
                signal.raw_percentage
                == signal.filtered_percentage
            )

        if signal.abrupt_change_confirmed:

            confirmed_count += 1

        serialised = signal.to_dict()

        json.dumps(
            serialised,
            allow_nan=False,
        )

        signals.append(serialised)

    summary = {
        "report": str(path),
        "observations": len(signals),
        "valid_signals": valid_count,
        "unavailable_signals": (
            len(signals) - valid_count
        ),
        "filter_modes": dict(modes),
        "abrupt_measurements": abrupt_count,
        "corroboration_observations": (
            confirmed_count
        ),
        "mean_absolute_residual_pp": (
            round(
                sum(residuals)
                / len(residuals),
                4,
            )
            if residuals else None
        ),
        "maximum_absolute_residual_pp": (
            round(max(residuals), 4)
            if residuals else None
        ),
        "mean_filter_ms": round(
            sum(processing_times)
            / max(len(processing_times), 1),
            4,
        ),
        "maximum_filter_ms": round(
            max(processing_times, default=0.0),
            4,
        ),
        "passed": True,
    }

    output_directory = Path(
        "data/validation/signal_filter"
    ) / path.parent.name

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory / "report.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "signals": signals,
            },
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n========== SIGNAL FILTER REPLAY ==========\n"
    )

    print(json.dumps(summary, indent=4))

    print(f"\nSaved to: {output_path}")

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        default=[],
    )

    args = parser.parse_args()

    run_synthetic_tests()

    reports = args.report

    if not reports:

        reports = sorted(
            Path(
                "data/validation/"
                "continuous_perception"
            ).glob("*/report.json")
        )

    if not reports:

        print("No recorded reports found.")

        return

    summaries = [
        replay_report(path)
        for path in reports
    ]

    combined = {
        "reports_tested": len(summaries),
        "total_observations": sum(
            item["observations"]
            for item in summaries
        ),
        "valid_signals": sum(
            item["valid_signals"]
            for item in summaries
        ),
        "unavailable_signals": sum(
            item["unavailable_signals"]
            for item in summaries
        ),
        "total_abrupt_measurements": sum(
            item["abrupt_measurements"]
            for item in summaries
        ),
        "total_corroboration_observations": sum(
            item["corroboration_observations"]
            for item in summaries
        ),
        "passed": all(
            item["passed"]
            for item in summaries
        ),
    }

    print(
        "\n========== COMBINED SUMMARY ==========\n"
    )

    print(json.dumps(combined, indent=4))

    print("\nAll signal filter tests passed.")


if __name__ == "__main__":

    main()
