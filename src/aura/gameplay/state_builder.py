
from __future__ import annotations

import math

from aura.gameplay.state import (
    Evidence,
    EvidenceSource,
    EvidenceStatus,
    GameplayState,
    ObservationStatus,
    PlayerState,
)

from aura.perception.continuous_perception import (
    PerceptionObservation,
)


class GameplayStateBuilder:
    """
    Converts perception observations into
    structured gameplay snapshots.

    This class does not retain historical state
    or perform temporal filtering.
    """

    def build(
        self,
        observation: PerceptionObservation,
    ) -> GameplayState:

        status = ObservationStatus(
            observation.status
        )

        health = self._build_health_evidence(
            observation
        )

        player = PlayerState(
            health_fraction=health,
        )

        return GameplayState(
            frame_id=observation.frame_id,
            timestamp=observation.timestamp,
            monotonic_timestamp=(
                observation.monotonic_timestamp
            ),
            observation_status=status,
            frame_available=(
                observation.frame_available
            ),
            player=player,
            reason=observation.reason,
        )

    def _build_health_evidence(
        self,
        observation: PerceptionObservation,
    ) -> Evidence[float]:

        if observation.status != "observed":

            return Evidence.unknown(
                reason=(
                    observation.reason
                    or observation.status
                )
            )

        if not observation.frame_available:

            return Evidence.unknown(
                reason="Frame unavailable."
            )

        if not observation.health_detected:

            return Evidence.unknown(
                reason="Health HUD not detected."
            )

        fraction = observation.health_fraction

        if fraction is None:

            return Evidence.unknown(
                reason="Health measurement missing."
            )

        if (
            not math.isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
        ):

            return Evidence.unknown(
                reason="Invalid health measurement."
            )

        return Evidence(
            value=float(fraction),
            status=EvidenceStatus.CURRENT,
            source=EvidenceSource.COMPUTER_VISION,
            timestamp=observation.timestamp,
            monotonic_timestamp=(
                observation.monotonic_timestamp
            ),
            quality=(
                observation.measurement_quality
            ),
        )
