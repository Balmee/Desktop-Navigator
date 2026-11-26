
from __future__ import annotations

import argparse
import json
import time

from collections import Counter
from pathlib import Path

from aura.gameplay.state import (
    EvidenceStatus,
)

from aura.gameplay.state_builder import (
    GameplayStateBuilder,
)

from aura.gameplay.temporal_tracker import (
    TemporalStateTracker,
    TemporalTrackerConfig,
)

from aura.perception.continuous_perception import (
    PerceptionObservation,
)


builder = GameplayStateBuilder()


def make_state(
    frame_id: int,
    timestamp: float,
    health: float | None = None,
    status: str = "observed",
):

    available = status != "unavailable"

    detected = (
        health is not None
        and status == "observed"
    )

    observation = PerceptionObservation(
        frame_id=frame_id,
        timestamp=1000.0 + timestamp,
        monotonic_timestamp=timestamp,
        status=status,
        frame_available=available,
        health_detected=detected,
        health_fraction=(
            health if detected else None
        ),
        health_percentage=(
            health * 100 if detected else None
        ),
        measurement_quality=(
            "provisional" if detected else "unknown"
        ),
        capture_ms=28.0,
        perception_ms=38.0,
        total_processing_ms=66.0,
    )

    return builder.build(observation)


def test_evidence_freshness():

    tracker = TemporalStateTracker()

    result = tracker.update(
        make_state(1, 0.0, 0.62)
    )

    assert result.current_health_percentage == 62.0

    assert (
        result.last_known_health.status
        == EvidenceStatus.CURRENT
    )

    result = tracker.update(
        make_state(
            2, 0.1,
            status="hud_unknown",
        )
    )

    assert result.current_health_percentage is None

    assert (
        result.last_known_health_percentage
        == 62.0
    )

    assert (
        result.last_known_health.status
        == EvidenceStatus.HISTORICAL
    )

    result = tracker.update(
        make_state(
            3, 0.8,
            status="hud_unknown",
        )
    )

    assert (
        result.last_known_health.status
        == EvidenceStatus.STALE
    )

    result = tracker.update(
        make_state(
            4, 2.1,
            status="hud_unknown",
        )
    )

    assert (
        result.last_known_health.status
        == EvidenceStatus.UNKNOWN
    )

    assert (
        result.last_known_health_percentage
        is None
    )

    result = tracker.update(
        make_state(5, 2.2, 0.38)
    )

    assert result.current_health_percentage == 38.0

    assert (
        result.last_known_health.status
        == EvidenceStatus.CURRENT
    )

    print("PASS: Evidence freshness lifecycle")


def test_declining_health():

    tracker = TemporalStateTracker()

    for i in range(51):

        timestamp = i / 10

        health = 1.0 - timestamp * 0.1

        result = tracker.update(
            make_state(
                i + 1,
                timestamp,
                health,
            )
        )

    for trend in result.health_trends:

        assert trend.available

        assert abs(
            trend.rate_pp_per_second + 10.0
        ) < 0.001

    print("PASS: Declining health trends")


def test_recovering_health():

    tracker = TemporalStateTracker()

    for i in range(51):

        timestamp = i / 10

        health = 0.2 + timestamp * 0.1

        result = tracker.update(
            make_state(
                i + 1,
                timestamp,
                health,
            )
        )

    for trend in result.health_trends:

        assert trend.available

        assert abs(
            trend.rate_pp_per_second - 10.0
        ) < 0.001

    print("PASS: Recovering health trends")


def test_stable_health():

    tracker = TemporalStateTracker()

    for i in range(51):

        result = tracker.update(
            make_state(
                i + 1,
                i / 10,
                0.75,
            )
        )

    for trend in result.health_trends:

        assert trend.available
        assert trend.change_pp == 0.0
        assert trend.rate_pp_per_second == 0.0

    print("PASS: Stable health trends")


def test_measurement_gap():

    tracker = TemporalStateTracker()

    for i in range(21):

        result = tracker.update(
            make_state(
                i + 1,
                i / 10,
                0.8,
            )
        )

    previous_continuity = result.continuity_id

    result = tracker.update(
        make_state(
            22, 2.8,
            status="hud_unknown",
        )
    )

    assert (
        result.continuity_id
        > previous_continuity
    )

    result = tracker.update(
        make_state(23, 3.0, 0.3)
    )

    assert result.current_health_percentage == 30.0

    assert not any(
        trend.available
        for trend in result.health_trends
    )

    print("PASS: Long measurement gap rejection")


def test_short_measurement_gap():

    tracker = TemporalStateTracker()

    for i in range(21):

        result = tracker.update(
            make_state(
                i + 1,
                i / 10,
                0.8,
            )
        )

    previous_continuity = result.continuity_id

    result = tracker.update(
        make_state(
            22, 2.1,
            status="hud_unknown",
        )
    )

    assert not any(
        trend.available
        for trend in result.health_trends
    )

    result = tracker.update(
        make_state(23, 2.2, 0.8)
    )

    assert (
        result.continuity_id
        == previous_continuity
    )

    assert result.health_trends[0].available

    print("PASS: Short measurement gap recovery")


def test_foreground_loss():

    tracker = TemporalStateTracker()

    tracker.update(
        make_state(1, 0.0, 0.7)
    )

    result = tracker.update(
        make_state(
            2, 0.1,
            status="unavailable",
        )
    )

    assert not result.observation_available

    assert result.current_health_percentage is None

    assert (
        result.last_known_health_percentage
        is None
    )

    assert not any(
        trend.available
        for trend in result.health_trends
    )

    result = tracker.update(
        make_state(3, 0.2, 0.4)
    )

    assert result.observation_available

    assert result.current_health_percentage == 40.0

    assert not any(
        trend.available
        for trend in result.health_trends
    )

    print("PASS: Foreground loss and reacquisition")


def test_bounded_history():

    tracker = TemporalStateTracker(
        TemporalTrackerConfig(
            history_seconds=5.0,
            max_observations=100,
        )
    )

    for i in range(201):

        result = tracker.update(
            make_state(
                i + 1,
                i / 10,
                0.5,
            )
        )

    assert result.history_size <= 51

    assert len(tracker.history) <= 100

    assert (
        tracker.history[-1].monotonic_timestamp
        - tracker.history[0].monotonic_timestamp
        <= 5.0
    )

    print("PASS: Bounded rolling history")


def test_order_validation():

    tracker = TemporalStateTracker()

    tracker.update(
        make_state(1, 1.0, 0.5)
    )

    try:

        tracker.update(
            make_state(2, 0.5, 0.4)
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Out-of-order timestamp accepted."
        )

    try:

        tracker.update(
            make_state(1, 2.0, 0.4)
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Duplicate frame ID accepted."
        )

    print("PASS: Observation ordering")


def test_session_reset():

    tracker = TemporalStateTracker()

    tracker.update(
        make_state(1, 10.0, 0.9)
    )

    tracker.reset()

    result = tracker.update(
        make_state(
            1, 0.0,
            status="hud_unknown",
        )
    )

    assert result.history_size == 1

    assert (
        result.last_known_health_percentage
        is None
    )

    assert result.continuity_id == 0

    print("PASS: Session reset")


def test_zero_health():

    tracker = TemporalStateTracker()

    result = tracker.update(
        make_state(1, 0.0, 0.0)
    )

    assert result.current_health_percentage == 0.0

    assert (
        result.last_known_health_percentage
        == 0.0
    )

    assert (
        result.last_known_health.status
        == EvidenceStatus.CURRENT
    )

    print("PASS: Zero health remains valid")


def run_synthetic_tests():

    print(
        "\n========== SYNTHETIC TESTS ==========\n"
    )

    tests = [
        test_evidence_freshness,
        test_declining_health,
        test_recovering_health,
        test_stable_health,
        test_measurement_gap,
        test_short_measurement_gap,
        test_foreground_loss,
        test_bounded_history,
        test_order_validation,
        test_session_reset,
        test_zero_health,
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

    tracker = TemporalStateTracker()

    counts = Counter()
    trend_counts = Counter()

    processing_times = []

    maximum_history_size = 0

    continuity_breaks = 0
    previous_continuity = 0

    example_trends = None

    valid_health_count = 0

    for raw in report["observations"]:

        observation = PerceptionObservation(
            **raw
        )

        state = builder.build(observation)

        start = time.perf_counter()

        result = tracker.update(state)

        processing_times.append(
            (time.perf_counter() - start)
            * 1000
        )

        counts[
            result.last_known_health.status.value
        ] += 1

        if state.health_available:

            valid_health_count += 1

            assert (
                result.current_health_percentage
                == state.health_percentage
            )

        else:

            assert (
                result.current_health_percentage
                is None
            )

        if not result.observation_available:

            assert (
                result.last_known_health.value
                is None
            )

        if (
            result.continuity_id
            != previous_continuity
        ):

            continuity_breaks += 1

        previous_continuity = (
            result.continuity_id
        )

        maximum_history_size = max(
            maximum_history_size,
            result.history_size,
        )

        for trend in result.health_trends:

            if trend.available:

                trend_counts[
                    f"{trend.window_seconds:g}s"
                ] += 1

                assert state.health_available

        if all(
            trend.available
            for trend in result.health_trends
        ):

            example_trends = [
                {
                    "window_seconds": (
                        trend.window_seconds
                    ),
                    "sample_count": (
                        trend.sample_count
                    ),
                    "change_pp": trend.change_pp,
                    "rate_pp_per_second": (
                        trend.rate_pp_per_second
                    ),
                }
                for trend in result.health_trends
            ]

        json.dumps(
            result.to_dict(),
            allow_nan=False,
        )

    summary = {
        "report": str(path),
        "observations": len(
            report["observations"]
        ),
        "valid_current_health_states": (
            valid_health_count
        ),
        "evidence_status_counts": dict(counts),
        "available_trend_counts": dict(
            trend_counts
        ),
        "continuity_breaks": continuity_breaks,
        "maximum_history_size": (
            maximum_history_size
        ),
        "mean_tracking_ms": round(
            sum(processing_times)
            / max(len(processing_times), 1),
            4,
        ),
        "maximum_tracking_ms": round(
            max(processing_times, default=0.0),
            4,
        ),
        "example_trends": example_trends,
        "passed": True,
    }

    print(
        "\n========== TEMPORAL REPLAY SUMMARY ==========\n"
    )

    print(json.dumps(summary, indent=4))

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
        "valid_current_health_states": sum(
            item["valid_current_health_states"]
            for item in summaries
        ),
        "total_continuity_breaks": sum(
            item["continuity_breaks"]
            for item in summaries
        ),
        "maximum_history_size": max(
            item["maximum_history_size"]
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

    print("\nAll temporal tracker tests passed.")


if __name__ == "__main__":

    main()
