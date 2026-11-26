
from __future__ import annotations

import argparse
import json
import time

from collections import Counter
from dataclasses import replace
from pathlib import Path

from aura.gameplay.state import (
    EvidenceStatus,
    EvidenceSource,
)

from aura.gameplay.state_builder import (
    GameplayStateBuilder,
)

from aura.perception.continuous_perception import (
    PerceptionObservation,
)


def test_edge_cases():

    builder = GameplayStateBuilder()

    observation = PerceptionObservation(
        frame_id=1,
        timestamp=1000.0,
        monotonic_timestamp=100.0,
        status="observed",
        frame_available=True,
        health_detected=True,
        health_fraction=0.62,
        health_percentage=62.0,
        measurement_quality="provisional",
        capture_ms=28.0,
        perception_ms=38.0,
        total_processing_ms=66.0,
    )

    # Test 1: Valid health.

    state = builder.build(observation)

    assert state.health_available
    assert state.health_percentage == 62.0

    assert (
        state.player.health_fraction.source
        == EvidenceSource.COMPUTER_VISION
    )

    # Test 2: Unknown health.

    unknown = replace(
        observation,
        status="hud_unknown",
        health_detected=False,
        health_fraction=None,
        health_percentage=None,
    )

    state = builder.build(unknown)

    assert not state.health_available
    assert state.health_percentage is None

    # Test 3: Unavailable game.

    unavailable = replace(
        unknown,
        status="unavailable",
        frame_available=False,
    )

    state = builder.build(unavailable)

    assert not state.frame_available
    assert state.health_percentage is None

    # Test 4: Zero health is not unknown.

    zero = replace(
        observation,
        health_fraction=0.0,
        health_percentage=0.0,
    )

    state = builder.build(zero)

    assert state.health_available
    assert state.health_percentage == 0.0

    # Test 5: Invalid numeric values.

    for value in (
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ):

        invalid = replace(
            observation,
            health_fraction=value,
        )

        state = builder.build(invalid)

        assert not state.health_available
        assert state.health_percentage is None

    # Test 6: Unobserved gameplay properties.

    state = builder.build(observation)

    assert state.player.hero.value is None
    assert state.player.alive.value is None
    assert state.resources.currency.value is None
    assert state.entities.value is None
    assert state.objectives.value is None

    assert (
        state.combat.active_engagement.status
        == EvidenceStatus.UNKNOWN
    )

    # Test 7: JSON serialisation.

    json.dumps(
        state.to_dict(),
        allow_nan=False,
    )

    print("All synthetic edge-case tests passed.")


def replay_report(path: Path):

    report = json.loads(
        path.read_text(encoding="utf-8")
    )

    builder = GameplayStateBuilder()

    counts = Counter()

    observations = report["observations"]

    states = []

    start = time.perf_counter()

    for raw in observations:

        observation = PerceptionObservation(
            **raw
        )

        state = builder.build(observation)

        states.append(state)

        counts[
            state.observation_status.value
        ] += 1

        assert (
            state.frame_id
            == observation.frame_id
        )

        assert (
            state.frame_available
            == observation.frame_available
        )

        assert (
            state.monotonic_timestamp
            == observation.monotonic_timestamp
        )

        if observation.status == "observed":

            assert state.health_available

            assert abs(
                state.health_percentage
                - observation.health_percentage
            ) <= 0.01

        else:

            assert not state.health_available

            assert (
                state.health_percentage is None
            )

        assert state.entities.value is None

        assert state.player.alive.value is None

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    summary = {
        "report": str(path),
        "observations": len(observations),
        "states_created": len(states),
        "status_counts": dict(counts),
        "valid_health_states": sum(
            state.health_available
            for state in states
        ),
        "unknown_health_states": sum(
            not state.health_available
            for state in states
        ),
        "total_build_ms": round(
            elapsed_ms, 3
        ),
        "mean_build_ms": (
            round(
                elapsed_ms / len(states),
                4,
            )
            if states else None
        ),
        "passed": True,
    }

    print("\n========== REPLAY SUMMARY ==========\n")

    print(json.dumps(summary, indent=4))

    valid_states = [
        state for state in states
        if state.health_available
    ]

    if valid_states:

        print("\n========== EXAMPLE STATE ==========\n")

        print(
            json.dumps(
                valid_states[0].to_dict(),
                indent=4,
                allow_nan=False,
            )
        )

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

    test_edge_cases()

    reports = args.report

    if not reports:

        reports = sorted(
            Path(
                "data/validation/"
                "continuous_perception"
            ).glob("*/report.json")
        )

    if not reports:

        print(
            "No recorded perception reports found."
        )

        return

    summaries = []

    for path in reports:

        summaries.append(
            replay_report(path)
        )

    combined = {
        "reports_tested": len(summaries),
        "total_observations": sum(
            item["observations"]
            for item in summaries
        ),
        "total_states_created": sum(
            item["states_created"]
            for item in summaries
        ),
        "valid_health_states": sum(
            item["valid_health_states"]
            for item in summaries
        ),
        "unknown_health_states": sum(
            item["unknown_health_states"]
            for item in summaries
        ),
        "passed": all(
            item["passed"]
            for item in summaries
        ),
    }

    print("\n========== COMBINED SUMMARY ==========\n")

    print(json.dumps(combined, indent=4))

    print("\nAll gameplay state tests passed.")


if __name__ == "__main__":
    main()
