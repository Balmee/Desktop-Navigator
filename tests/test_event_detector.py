
from __future__ import annotations

import argparse
import json
import time

from collections import Counter
from pathlib import Path

from aura.gameplay.state_builder import (
    GameplayStateBuilder,
)

from aura.gameplay.temporal_tracker import (
    TemporalStateTracker,
)

from aura.gameplay.signal_filter import (
    HealthSignalFilter,
)

from aura.gameplay.event_detector import (
    GameplayEventDetector,
)

from aura.gameplay.events import (
    EventType,
    HealthBand,
)

from aura.perception.continuous_perception import (
    PerceptionObservation,
)


class TestPipeline:

    def __init__(self):

        self.builder = GameplayStateBuilder()

        self.tracker = TemporalStateTracker()

        self.filter = HealthSignalFilter()

        self.detector = GameplayEventDetector()

        self.frame_id = 0

        self.events = []

    def update(
        self,
        timestamp,
        health,
        status="observed",
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
            frame_available=status != "unavailable",
            health_detected=detected,
            health_fraction=(
                health / 100 if detected else None
            ),
            health_percentage=(
                health if detected else None
            ),
            measurement_quality=(
                "provisional" if detected else "unknown"
            ),
            capture_ms=28.0,
            perception_ms=38.0,
            total_processing_ms=66.0,
        )

        state = self.builder.build(observation)

        temporal = self.tracker.update(state)

        signal = self.filter.update(temporal)

        result = self.detector.update(
            temporal,
            signal,
        )

        self.events.extend(result.events)

        return result

    def count(self, event_type):

        return sum(
            event.event_type == event_type
            for event in self.events
        )


def test_stable_health():

    pipeline = TestPipeline()

    for i in range(101):

        pipeline.update(i / 10, 80.0)

    assert not pipeline.events

    print("PASS: Stable health produces no events")


def test_rapid_loss():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    result = pipeline.update(0.1, 30.0)

    assert result.pending_abrupt_change

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 0

    pipeline.update(0.2, 30.5)

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 1

    event = next(
        event for event in pipeline.events
        if event.event_type == EventType.RAPID_HEALTH_LOSS
    )

    assert event.corroborated

    assert event.change_pp < 0

    print("PASS: Corroborated rapid health loss")


def test_rapid_gain():

    pipeline = TestPipeline()

    pipeline.update(0.0, 30.0)

    pipeline.update(0.1, 80.0)

    pipeline.update(0.2, 80.0)

    assert pipeline.count(
        EventType.RAPID_HEALTH_GAIN
    ) == 1

    print("PASS: Corroborated rapid health gain")


def test_transient_spike():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(0.1, 10.0)

    pipeline.update(0.2, 80.0)

    pipeline.update(0.3, 80.0)

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 0

    assert pipeline.count(
        EventType.RAPID_HEALTH_GAIN
    ) == 0

    assert pipeline.count(
        EventType.CRITICAL_HEALTH_ENTERED
    ) == 0

    print("PASS: Transient spike does not trigger events")


def test_low_health():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)
    pipeline.update(0.1, 80.0)

    pipeline.update(0.2, 35.0)
    pipeline.update(0.3, 35.0)

    result = pipeline.update(0.4, 35.0)

    assert result.health_band == HealthBand.LOW

    assert pipeline.count(
        EventType.LOW_HEALTH_ENTERED
    ) == 1

    print("PASS: Low-health entry")


def test_critical_health():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)
    pipeline.update(0.1, 80.0)

    pipeline.update(0.2, 15.0)
    pipeline.update(0.3, 15.0)

    result = pipeline.update(0.4, 15.0)

    assert result.health_band == HealthBand.CRITICAL

    assert pipeline.count(
        EventType.CRITICAL_HEALTH_ENTERED
    ) == 1

    print("PASS: Critical-health entry")


def test_hysteresis():

    pipeline = TestPipeline()

    pipeline.update(0.0, 18.0)
    pipeline.update(0.1, 18.0)

    for i in range(2, 20):

        result = pipeline.update(
            i / 10,
            21.0 + (i % 2),
        )

    assert result.health_band == HealthBand.CRITICAL

    assert pipeline.count(
        EventType.CRITICAL_HEALTH_ENTERED
    ) == 1

    assert pipeline.count(
        EventType.CRITICAL_HEALTH_EXITED
    ) == 0

    pipeline.update(2.0, 30.0)
    pipeline.update(2.1, 30.0)

    result = pipeline.update(2.2, 30.0)

    assert result.health_band == HealthBand.LOW

    assert pipeline.count(
        EventType.CRITICAL_HEALTH_EXITED
    ) == 1

    print("PASS: Threshold hysteresis")


def test_low_health_exit():

    pipeline = TestPipeline()

    pipeline.update(0.0, 35.0)
    pipeline.update(0.1, 35.0)

    pipeline.update(0.2, 55.0)
    pipeline.update(0.3, 55.0)

    result = pipeline.update(0.4, 55.0)

    assert result.health_band == HealthBand.NORMAL

    assert pipeline.count(
        EventType.LOW_HEALTH_EXITED
    ) == 1

    print("PASS: Low-health recovery")


def test_sustained_decline():

    pipeline = TestPipeline()

    for i in range(61):

        pipeline.update(
            i / 10,
            90.0 - i * 0.7,
        )

    assert pipeline.count(
        EventType.SUSTAINED_HEALTH_DECLINE
    ) == 1

    print("PASS: Sustained health decline")


def test_sustained_recovery():

    pipeline = TestPipeline()

    for i in range(61):

        pipeline.update(
            i / 10,
            30.0 + i * 0.6,
        )

    assert pipeline.count(
        EventType.SUSTAINED_HEALTH_RECOVERY
    ) == 1

    print("PASS: Sustained health recovery")


def test_observation_loss():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(
        0.1, None, "unavailable"
    )

    pipeline.update(
        0.2, None, "unavailable"
    )

    pipeline.update(0.3, 30.0)

    assert pipeline.count(
        EventType.OBSERVATION_LOST
    ) == 1

    assert pipeline.count(
        EventType.OBSERVATION_RESTORED
    ) == 1

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 0

    print("PASS: Observation loss and restoration")


def test_health_signal_loss():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(
        0.1, None, "hud_unknown"
    )

    pipeline.update(
        0.2, None, "hud_unknown"
    )

    pipeline.update(0.3, 80.0)

    assert pipeline.count(
        EventType.HEALTH_SIGNAL_LOST
    ) == 1

    assert pipeline.count(
        EventType.HEALTH_SIGNAL_RESTORED
    ) == 1

    print("PASS: Health signal loss and restoration")


def test_missing_abrupt_confirmation():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(0.1, 20.0)

    pipeline.update(
        0.2, None, "hud_unknown"
    )

    pipeline.update(0.3, 20.0)

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 0

    print("PASS: Missing sample invalidates corroboration")


def test_zero_health():

    pipeline = TestPipeline()

    pipeline.update(0.0, 10.0)

    for i in range(1, 10):

        pipeline.update(i / 10, 0.0)

    assert pipeline.count(
        EventType.ZERO_HEALTH_OBSERVED
    ) == 1

    print("PASS: Corroborated zero-health observation")


def test_long_gap():

    pipeline = TestPipeline()

    pipeline.update(0.0, 80.0)

    pipeline.update(
        1.0, None, "hud_unknown"
    )

    pipeline.update(1.1, 20.0)

    pipeline.update(1.2, 20.0)

    assert pipeline.count(
        EventType.RAPID_HEALTH_LOSS
    ) == 0

    print("PASS: No rapid event across long gap")


def run_synthetic_tests():

    print(
        "\n========== SYNTHETIC TESTS ==========\n"
    )

    tests = [
        test_stable_health,
        test_rapid_loss,
        test_rapid_gain,
        test_transient_spike,
        test_low_health,
        test_critical_health,
        test_hysteresis,
        test_low_health_exit,
        test_sustained_decline,
        test_sustained_recovery,
        test_observation_loss,
        test_health_signal_loss,
        test_missing_abrupt_confirmation,
        test_zero_health,
        test_long_gap,
    ]

    for test in tests:

        test()

    print(
        f"\nAll {len(tests)} synthetic tests passed."
    )


def replay_report(path):

    report = json.loads(
        path.read_text(encoding="utf-8")
    )

    builder = GameplayStateBuilder()

    tracker = TemporalStateTracker()

    signal_filter = HealthSignalFilter()

    detector = GameplayEventDetector()

    events = []

    counts = Counter()
    severities = Counter()
    health_bands = Counter()

    processing_times = []

    pending_count = 0

    for raw in report["observations"]:

        observation = PerceptionObservation(**raw)

        state = builder.build(observation)

        temporal = tracker.update(state)

        signal = signal_filter.update(temporal)

        start = time.perf_counter()

        result = detector.update(
            temporal,
            signal,
        )

        processing_times.append(
            (time.perf_counter() - start) * 1000
        )

        health_bands[result.health_band.value] += 1

        if result.pending_abrupt_change:

            pending_count += 1

        for event in result.events:

            counts[event.event_type.value] += 1

            severities[event.severity.value] += 1

            if event.event_type in (
                EventType.RAPID_HEALTH_LOSS,
                EventType.RAPID_HEALTH_GAIN,
            ):

                assert event.corroborated

                assert result.health_available

                assert (
                    event.originating_frame_id
                    < event.frame_id
                )

            if not result.observation_available:

                assert (
                    event.event_type
                    == EventType.OBSERVATION_LOST
                )

            events.append(event.to_dict())

        json.dumps(
            result.to_dict(),
            allow_nan=False,
        )

    event_ids = [
        event["event_id"]
        for event in events
    ]

    assert event_ids == list(
        range(1, len(events) + 1)
    )

    summary = {
        "report": str(path),
        "observations": len(report["observations"]),
        "events_detected": len(events),
        "event_counts": dict(counts),
        "severity_counts": dict(severities),
        "health_band_counts": dict(health_bands),
        "pending_abrupt_observations": pending_count,
        "mean_event_detection_ms": round(
            sum(processing_times)
            / max(len(processing_times), 1),
            4,
        ),
        "maximum_event_detection_ms": round(
            max(processing_times, default=0.0),
            4,
        ),
        "passed": True,
    }

    output_directory = (
        Path("data/validation/event_detection")
        / path.parent.name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / "report.json"

    output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "events": events,
            },
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n========== EVENT DETECTION REPLAY ==========\n"
    )

    print(json.dumps(summary, indent=4))

    print("\nDetected events:")

    for event in events:

        print(
            f"[Frame {event['frame_id']}] "
            f"{event['severity'].value.upper()} "
            f"{event['event_type'].value} | "
            f"Health: {event['current_health_percentage']}"
        )

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
                "data/validation/continuous_perception"
            ).glob("*/report.json")
        )

    if not reports:

        print("No recorded reports found.")

        return

    summaries = [
        replay_report(path)
        for path in reports
    ]

    event_counts = Counter()

    severity_counts = Counter()

    for summary in summaries:

        event_counts.update(
            summary["event_counts"]
        )

        severity_counts.update(
            summary["severity_counts"]
        )

    combined = {
        "reports_tested": len(summaries),
        "total_observations": sum(
            item["observations"]
            for item in summaries
        ),
        "total_events": sum(
            item["events_detected"]
            for item in summaries
        ),
        "event_counts": dict(event_counts),
        "severity_counts": dict(severity_counts),
        "passed": all(
            item["passed"]
            for item in summaries
        ),
    }

    print(
        "\n========== COMBINED SUMMARY ==========\n"
    )

    print(json.dumps(combined, indent=4))

    print("\nAll gameplay event tests passed.")


if __name__ == "__main__":

    main()
