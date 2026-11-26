
from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass, asdict, replace

from aura.gameplay.state import (
    Evidence,
    EvidenceStatus,
    GameplayState,
    ObservationStatus,
)


@dataclass(frozen=True)
class TemporalTrackerConfig:

    history_seconds: float = 30.0
    max_observations: int = 600

    historical_seconds: float = 0.5
    expiration_seconds: float = 2.0

    maximum_measurement_gap: float = 0.5

    trend_windows: tuple[float, ...] = (
        1.0,
        3.0,
        5.0,
    )

    def __post_init__(self):

        durations = (
            self.history_seconds,
            self.historical_seconds,
            self.expiration_seconds,
            self.maximum_measurement_gap,
            *self.trend_windows,
        )

        if any(
            not math.isfinite(value)
            or value <= 0
            for value in durations
        ):
            raise ValueError(
                "Durations must be finite and positive."
            )

        if self.max_observations < 2:
            raise ValueError(
                "At least two observations are required."
            )

        if (
            self.historical_seconds
            >= self.expiration_seconds
        ):
            raise ValueError(
                "Historical duration must be shorter "
                "than expiration duration."
            )

        if any(
            window > self.history_seconds
            for window in self.trend_windows
        ):
            raise ValueError(
                "Trend windows cannot exceed history duration."
            )


@dataclass(frozen=True)
class HealthTrend:

    window_seconds: float

    available: bool = False

    sample_count: int = 0
    observed_duration_seconds: float = 0.0

    starting_percentage: float | None = None
    ending_percentage: float | None = None

    minimum_percentage: float | None = None
    maximum_percentage: float | None = None

    change_pp: float | None = None
    rate_pp_per_second: float | None = None

    reason: str | None = None


@dataclass(frozen=True)
class TemporalGameplayState:

    current: GameplayState

    last_known_health: Evidence[float]

    last_known_health_age_seconds: float | None

    health_trends: tuple[HealthTrend, ...]

    history_size: int

    consecutive_valid_observations: int
    consecutive_unknown_observations: int

    continuity_id: int

    observation_available: bool

    @property
    def current_health_percentage(
        self,
    ) -> float | None:

        return self.current.health_percentage

    @property
    def last_known_health_percentage(
        self,
    ) -> float | None:

        value = self.last_known_health.value

        if value is None:
            return None

        return round(value * 100, 2)

    def to_dict(self) -> dict:

        result = asdict(self)

        result["current_health_percentage"] = (
            self.current_health_percentage
        )

        result["last_known_health_percentage"] = (
            self.last_known_health_percentage
        )

        return result


class TemporalStateTracker:
    """
    Maintains bounded gameplay history and tracks
    the freshness and continuity of health evidence.

    Does not perform signal smoothing, event
    classification or gameplay coaching.
    """

    def __init__(
        self,
        config: TemporalTrackerConfig | None = None,
    ):

        self.config = (
            config or TemporalTrackerConfig()
        )

        self.reset()

    def reset(self) -> None:

        self._history = deque(
            maxlen=self.config.max_observations
        )

        self._health_history = deque(
            maxlen=self.config.max_observations
        )

        self._last_known_health = (
            Evidence[float].unknown()
        )

        self._last_timestamp: float | None = None
        self._last_frame_id: int | None = None

        self._last_valid_timestamp: float | None = None

        self._consecutive_valid = 0
        self._consecutive_unknown = 0

        self._continuity_id = 0

        self._observation_available = False

    @property
    def history(self) -> tuple[GameplayState, ...]:

        return tuple(self._history)

    def update(
        self,
        state: GameplayState,
    ) -> TemporalGameplayState:

        now = state.monotonic_timestamp

        self._validate_order(state)

        self._history.append(state)

        self._prune_history(now)

        available = (
            state.frame_available
            and state.observation_status
            != ObservationStatus.UNAVAILABLE
            and state.observation_status
            != ObservationStatus.CAPTURE_ERROR
        )

        if not available:

            self._handle_unavailable()

        elif state.health_available:

            self._handle_valid(state)

        else:

            self._handle_unknown(now)

        self._observation_available = available

        self._last_timestamp = now
        self._last_frame_id = state.frame_id

        last_known, age = (
            self._get_last_known_health(now)
        )

        trends = tuple(
            self._calculate_trend(
                now=now,
                window=window,
                current_available=(
                    available
                    and state.health_available
                ),
            )
            for window in self.config.trend_windows
        )

        return TemporalGameplayState(
            current=state,
            last_known_health=last_known,
            last_known_health_age_seconds=age,
            health_trends=trends,
            history_size=len(self._history),
            consecutive_valid_observations=(
                self._consecutive_valid
            ),
            consecutive_unknown_observations=(
                self._consecutive_unknown
            ),
            continuity_id=self._continuity_id,
            observation_available=available,
        )

    def _validate_order(
        self,
        state: GameplayState,
    ) -> None:

        timestamp = state.monotonic_timestamp

        if not math.isfinite(timestamp):

            raise ValueError(
                "Invalid monotonic timestamp."
            )

        if (
            self._last_timestamp is not None
            and timestamp < self._last_timestamp
        ):

            raise ValueError(
                "Observations must arrive in "
                "chronological order."
            )

        if (
            self._last_frame_id is not None
            and state.frame_id <= self._last_frame_id
        ):

            raise ValueError(
                "Frame IDs must increase. "
                "Reset the tracker for a new session."
            )

    def _prune_history(
        self,
        now: float,
    ) -> None:

        cutoff = (
            now - self.config.history_seconds
        )

        while (
            self._history
            and self._history[0].monotonic_timestamp
            < cutoff
        ):

            self._history.popleft()

        while (
            self._health_history
            and self._health_history[0][0] < cutoff
        ):

            self._health_history.popleft()

    def _break_continuity(self) -> None:

        if self._last_valid_timestamp is not None:

            self._continuity_id += 1

        self._health_history.clear()

        self._last_valid_timestamp = None

    def _handle_unavailable(self) -> None:

        self._break_continuity()

        self._last_known_health = (
            Evidence.unknown(
                reason="Gameplay observation unavailable."
            )
        )

        self._consecutive_valid = 0
        self._consecutive_unknown += 1

    def _handle_valid(
        self,
        state: GameplayState,
    ) -> None:

        now = state.monotonic_timestamp

        if self._last_valid_timestamp is not None:

            gap = now - self._last_valid_timestamp

            if gap > self.config.maximum_measurement_gap:

                self._break_continuity()

        evidence = state.player.health_fraction

        value = evidence.value

        if value is None:
            return

        self._health_history.append(
            (now, float(value))
        )

        self._last_known_health = evidence

        self._last_valid_timestamp = now

        self._consecutive_valid += 1
        self._consecutive_unknown = 0

    def _handle_unknown(
        self,
        now: float,
    ) -> None:

        self._consecutive_valid = 0
        self._consecutive_unknown += 1

        if self._last_valid_timestamp is not None:

            gap = now - self._last_valid_timestamp

            if gap > self.config.maximum_measurement_gap:

                self._break_continuity()

    def _get_last_known_health(
        self,
        now: float,
    ) -> tuple[Evidence[float], float | None]:

        evidence = self._last_known_health

        if (
            evidence.value is None
            or evidence.monotonic_timestamp is None
        ):

            return evidence, None

        age = max(
            0.0,
            now - evidence.monotonic_timestamp,
        )

        if self._consecutive_valid > 0:

            return evidence, age

        if age >= self.config.expiration_seconds:

            return (
                Evidence.unknown(
                    reason="Previous health evidence expired."
                ),
                None,
            )

        if age <= self.config.historical_seconds:

            status = EvidenceStatus.HISTORICAL

        else:

            status = EvidenceStatus.STALE

        return (
            replace(
                evidence,
                status=status,
                reason="No current health measurement.",
            ),
            age,
        )

    def _calculate_trend(
        self,
        now: float,
        window: float,
        current_available: bool,
    ) -> HealthTrend:

        if not current_available:

            return HealthTrend(
                window_seconds=window,
                reason="Current health unavailable.",
            )

        cutoff = now - window

        samples = [
            (timestamp, value)
            for timestamp, value in self._health_history
            if timestamp >= cutoff
        ]

        if len(samples) < 2:

            return HealthTrend(
                window_seconds=window,
                sample_count=len(samples),
                reason="Insufficient continuous samples.",
            )

        first_time, first_value = samples[0]
        last_time, last_value = samples[-1]

        duration = last_time - first_time

        if duration <= 0:

            return HealthTrend(
                window_seconds=window,
                sample_count=len(samples),
                reason="Insufficient elapsed time.",
            )

        # Require meaningful coverage of the requested
        # window before reporting a trend.

        if duration < window * 0.8:

            return HealthTrend(
                window_seconds=window,
                sample_count=len(samples),
                observed_duration_seconds=round(
                    duration, 4
                ),
                reason="Insufficient window coverage.",
            )

        values = [
            value * 100
            for _, value in samples
        ]

        change = (
            last_value - first_value
        ) * 100

        return HealthTrend(
            window_seconds=window,
            available=True,
            sample_count=len(samples),
            observed_duration_seconds=round(
                duration, 4
            ),
            starting_percentage=round(
                first_value * 100, 3
            ),
            ending_percentage=round(
                last_value * 100, 3
            ),
            minimum_percentage=round(
                min(values), 3
            ),
            maximum_percentage=round(
                max(values), 3
            ),
            change_pp=round(change, 4),
            rate_pp_per_second=round(
                change / duration, 4
            ),
        )
