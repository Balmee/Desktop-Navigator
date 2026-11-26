
from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass

from aura.gameplay.events import (
    EventType,
    EventSeverity,
    HealthBand,
    GameplayEvent,
    EventDetectionResult,
)

from aura.gameplay.temporal_tracker import (
    TemporalGameplayState,
)

from aura.gameplay.signal_filter import (
    FilteredHealthSignal,
)


@dataclass(frozen=True)
class EventDetectorConfig:

    low_health_threshold: float = 40.0
    critical_health_threshold: float = 20.0

    hysteresis_pp: float = 5.0

    rapid_change_threshold_pp: float = 5.0

    abrupt_confirmation_tolerance_pp: float = 2.0
    abrupt_confirmation_window: float = 0.5

    trend_window_seconds: float = 3.0

    decline_rate_threshold: float = 4.0
    recovery_rate_threshold: float = 3.0

    minimum_trend_change_pp: float = 8.0

    trend_cooldown_seconds: float = 5.0

    maximum_measurement_gap: float = 0.5

    confirmation_samples: int = 2

    def __post_init__(self):

        values = (
            self.low_health_threshold,
            self.critical_health_threshold,
            self.hysteresis_pp,
            self.rapid_change_threshold_pp,
            self.abrupt_confirmation_tolerance_pp,
            self.abrupt_confirmation_window,
            self.trend_window_seconds,
            self.decline_rate_threshold,
            self.recovery_rate_threshold,
            self.minimum_trend_change_pp,
            self.trend_cooldown_seconds,
            self.maximum_measurement_gap,
        )

        if any(
            not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError(
                "Event parameters must be finite and positive."
            )

        if not (
            0
            < self.critical_health_threshold
            < self.low_health_threshold
            < 100
        ):
            raise ValueError(
                "Invalid health thresholds."
            )

        if (
            self.critical_health_threshold
            + self.hysteresis_pp
            >= self.low_health_threshold
        ):
            raise ValueError(
                "Critical and low health bands overlap."
            )

        if (
            self.low_health_threshold
            + self.hysteresis_pp
            > 100
        ):
            raise ValueError(
                "Low health exit exceeds 100%."
            )

        if self.confirmation_samples < 2:

            raise ValueError(
                "At least two confirmation samples required."
            )


@dataclass(frozen=True)
class PendingAbruptChange:

    frame_id: int
    timestamp: float

    previous_health: float
    observed_health: float

    change_pp: float


class GameplayEventDetector:
    """
    Detects evidence-backed gameplay events.

    Does not generate coaching advice or infer
    enemy intentions, damage sources or death.
    """

    def __init__(
        self,
        config: EventDetectorConfig | None = None,
    ):

        self.config = config or EventDetectorConfig()

        self.reset()

    def reset(self) -> None:

        self._event_id = 0

        self._last_frame_id = None
        self._last_timestamp = None

        self._continuity_id = None

        self._observation_available = None
        self._health_available = None

        self._health_band = HealthBand.UNKNOWN

        self._candidate_band = None
        self._candidate_count = 0

        self._pending_abrupt = None

        self._previous_health = None
        self._last_valid_timestamp = None

        self._zero_count = 0
        self._zero_emitted = False

        self._trend_history = deque(maxlen=600)

        self._last_event_times = {}

        self._active_trend = None

    def _reset_health_context(self) -> None:

        self._health_band = HealthBand.UNKNOWN

        self._candidate_band = None
        self._candidate_count = 0

        self._pending_abrupt = None

        self._previous_health = None
        self._last_valid_timestamp = None

        self._zero_count = 0
        self._zero_emitted = False

        self._trend_history.clear()

        self._active_trend = None

        self._last_event_times.clear()

    def update(
        self,
        temporal: TemporalGameplayState,
        signal: FilteredHealthSignal,
    ) -> EventDetectionResult:

        self._validate_inputs(temporal, signal)

        events = []

        now = signal.monotonic_timestamp

        if (
            self._continuity_id is not None
            and signal.continuity_id
            != self._continuity_id
        ):

            self._reset_health_context()

        self._continuity_id = signal.continuity_id

        self._last_frame_id = signal.frame_id
        self._last_timestamp = now

        available = temporal.observation_available

        health_available = (
            available
            and signal.available
            and signal.filtered_percentage is not None
        )

        self._detect_availability(
            signal,
            available,
            health_available,
            events,
        )

        if not available:

            self._reset_health_context()

        elif not health_available:

            self._pending_abrupt = None

            self._candidate_band = None
            self._candidate_count = 0

            self._zero_count = 0

            if self._last_valid_timestamp is not None:

                gap = now - self._last_valid_timestamp

                if gap > self.config.maximum_measurement_gap:

                    self._reset_health_context()

        else:

            self._process_health(signal, events)

        self._observation_available = available
        self._health_available = health_available

        return EventDetectionResult(
            frame_id=signal.frame_id,
            monotonic_timestamp=now,
            continuity_id=signal.continuity_id,
            observation_available=available,
            health_available=health_available,
            health_band=(
                self._health_band
                if health_available
                else HealthBand.UNKNOWN
            ),
            pending_abrupt_change=(
                self._pending_abrupt is not None
            ),
            events=tuple(events),
        )

    def _validate_inputs(
        self,
        temporal,
        signal,
    ) -> None:

        if (
            temporal.current.frame_id
            != signal.frame_id
        ):

            raise ValueError(
                "Temporal and filtered frame IDs differ."
            )

        if (
            temporal.current.monotonic_timestamp
            != signal.monotonic_timestamp
        ):

            raise ValueError(
                "Temporal and filtered timestamps differ."
            )

        if temporal.continuity_id != signal.continuity_id:

            raise ValueError(
                "Continuity IDs differ."
            )

        if not math.isfinite(signal.monotonic_timestamp):

            raise ValueError(
                "Invalid observation timestamp."
            )

        if (
            self._last_frame_id is not None
            and signal.frame_id <= self._last_frame_id
        ):

            raise ValueError(
                "Frame IDs must increase."
            )

        if (
            self._last_timestamp is not None
            and signal.monotonic_timestamp
            < self._last_timestamp
        ):

            raise ValueError(
                "Observations must be chronological."
            )

        if signal.available:

            for value in (
                signal.raw_percentage,
                signal.filtered_percentage,
            ):

                if (
                    value is None
                    or not math.isfinite(value)
                    or not 0 <= value <= 100
                ):

                    raise ValueError(
                        "Invalid available health signal."
                    )

    def _emit(
        self,
        signal,
        events,
        event_type,
        severity,
        description,
        source="health_signal",
        **kwargs,
    ):

        self._event_id += 1

        event = GameplayEvent(
            event_id=self._event_id,
            event_type=event_type,
            severity=severity,
            frame_id=signal.frame_id,
            monotonic_timestamp=(
                signal.monotonic_timestamp
            ),
            continuity_id=signal.continuity_id,
            source=source,
            description=description,
            **kwargs,
        )

        events.append(event)

        self._last_event_times[event_type] = (
            signal.monotonic_timestamp
        )

    def _detect_availability(
        self,
        signal,
        available,
        health_available,
        events,
    ):

        previous = self._observation_available

        if previous is True and not available:

            self._emit(
                signal,
                events,
                EventType.OBSERVATION_LOST,
                EventSeverity.SYSTEM,
                "Gameplay observation became unavailable.",
                source="observation_status",
            )

        elif previous is False and available:

            self._emit(
                signal,
                events,
                EventType.OBSERVATION_RESTORED,
                EventSeverity.SYSTEM,
                "Gameplay observation resumed.",
                source="observation_status",
            )

        # Avoid duplicate health-loss notifications
        # when the entire observation is unavailable.

        if not available:

            return

        if previous is not True:

            return

        if (
            self._health_available is True
            and not health_available
        ):

            self._emit(
                signal,
                events,
                EventType.HEALTH_SIGNAL_LOST,
                EventSeverity.SYSTEM,
                "Current health measurement became unknown.",
            )

        elif (
            self._health_available is False
            and health_available
        ):

            self._emit(
                signal,
                events,
                EventType.HEALTH_SIGNAL_RESTORED,
                EventSeverity.SYSTEM,
                "Current health measurement resumed.",
            )

    def _process_health(
        self,
        signal,
        events,
    ):

        now = signal.monotonic_timestamp

        raw = signal.raw_percentage
        health = signal.filtered_percentage

        if self._last_valid_timestamp is not None:

            gap = now - self._last_valid_timestamp

            if gap > self.config.maximum_measurement_gap:

                self._reset_health_context()

        pending = self._pending_abrupt

        if pending is not None:

            self._pending_abrupt = None

            elapsed = now - pending.timestamp

            corroborated = (
                elapsed
                <= self.config.abrupt_confirmation_window
                and abs(
                    raw - pending.observed_health
                )
                <= self.config.abrupt_confirmation_tolerance_pp
            )

            if corroborated:

                self._emit_rapid_change(
                    signal,
                    events,
                    pending,
                )

                # Begin a fresh trend segment rather than
                # treating a sudden jump as a sustained trend.

                self._trend_history.clear()

            else:

                # The earlier abrupt sample is not
                # corroborated. Exclude it from trend history.

                self._trend_history.clear()

                self._candidate_band = None
                self._candidate_count = 0

                self._zero_count = 0

                # If the new measurement returns to the
                # previous baseline, treat it as a possible
                # transient spike rather than a new rapid gain.

                if abs(
                    raw - pending.previous_health
                ) <= self.config.abrupt_confirmation_tolerance_pp:

                    self._previous_health = health

                    self._last_valid_timestamp = now

                    self._update_health_band(
                        signal,
                        events,
                        health,
                    )

                    self._update_zero_health(
                        signal,
                        events,
                    )

                    self._update_trend(
                        signal,
                        events,
                        health,
                    )

                    return

        if (
            signal.abrupt_change
            and self._previous_health is not None
        ):

            baseline = (
                pending.previous_health
                if pending is not None
                and not corroborated
                else self._previous_health
            )

            change = raw - baseline

            if (
                abs(change)
                >= self.config.rapid_change_threshold_pp
            ):

                self._pending_abrupt = PendingAbruptChange(
                    frame_id=signal.frame_id,
                    timestamp=now,
                    previous_health=baseline,
                    observed_health=raw,
                    change_pp=change,
                )

                # Do not classify an unconfirmed abrupt
                # sample as a threshold or trend event.

                self._candidate_band = None
                self._candidate_count = 0

                self._zero_count = 0

                self._last_valid_timestamp = now

                return

        self._update_health_band(
            signal,
            events,
            health,
        )

        self._update_zero_health(
            signal,
            events,
        )

        self._update_trend(
            signal,
            events,
            health,
        )

        self._previous_health = health
        self._last_valid_timestamp = now

    def _emit_rapid_change(
        self,
        signal,
        events,
        pending,
    ):

        loss = pending.change_pp < 0

        event_type = (
            EventType.RAPID_HEALTH_LOSS
            if loss
            else EventType.RAPID_HEALTH_GAIN
        )

        severity = (
            EventSeverity.HIGH
            if loss
            else EventSeverity.MEDIUM
        )

        if (
            loss
            and signal.filtered_percentage
            <= self.config.critical_health_threshold
        ):

            severity = EventSeverity.CRITICAL

        self._emit(
            signal,
            events,
            event_type,
            severity,
            "Abrupt health change corroborated "
            "by a subsequent measurement.",
            corroborated=True,
            previous_health_percentage=(
                pending.previous_health
            ),
            current_health_percentage=(
                signal.filtered_percentage
            ),
            change_pp=round(
                signal.filtered_percentage
                - pending.previous_health,
                4,
            ),
            originating_frame_id=pending.frame_id,
        )

    def _classify_health_band(
        self,
        health,
    ) -> HealthBand:

        critical = self.config.critical_health_threshold
        low = self.config.low_health_threshold

        hysteresis = self.config.hysteresis_pp

        if health <= critical:

            return HealthBand.CRITICAL

        if (
            self._health_band == HealthBand.CRITICAL
            and health < critical + hysteresis
        ):

            return HealthBand.CRITICAL

        if health <= low:

            return HealthBand.LOW

        if (
            self._health_band in (
                HealthBand.LOW,
                HealthBand.CRITICAL,
            )
            and health < low + hysteresis
        ):

            return HealthBand.LOW

        return HealthBand.NORMAL

    def _update_health_band(
        self,
        signal,
        events,
        health,
    ):

        target = self._classify_health_band(health)

        if target == self._health_band:

            self._candidate_band = None
            self._candidate_count = 0

            return

        if target == self._candidate_band:

            self._candidate_count += 1

        else:

            self._candidate_band = target
            self._candidate_count = 1

        if (
            self._candidate_count
            < self.config.confirmation_samples
        ):

            return

        previous = self._health_band

        self._health_band = target

        self._candidate_band = None
        self._candidate_count = 0

        previously_low = previous in (
            HealthBand.LOW,
            HealthBand.CRITICAL,
        )

        currently_low = target in (
            HealthBand.LOW,
            HealthBand.CRITICAL,
        )

        if (
            previous == HealthBand.CRITICAL
            and target != HealthBand.CRITICAL
        ):

            self._emit(
                signal,
                events,
                EventType.CRITICAL_HEALTH_EXITED,
                EventSeverity.MEDIUM,
                "Health recovered above the critical band.",
                corroborated=True,
                current_health_percentage=health,
            )

        if previously_low and not currently_low:

            self._emit(
                signal,
                events,
                EventType.LOW_HEALTH_EXITED,
                EventSeverity.LOW,
                "Health recovered above the low-health band.",
                corroborated=True,
                current_health_percentage=health,
            )

        if target == HealthBand.CRITICAL:

            self._emit(
                signal,
                events,
                EventType.CRITICAL_HEALTH_ENTERED,
                EventSeverity.CRITICAL,
                "Health entered the critical band.",
                corroborated=True,
                current_health_percentage=health,
            )

        elif (
            target == HealthBand.LOW
            and not previously_low
        ):

            self._emit(
                signal,
                events,
                EventType.LOW_HEALTH_ENTERED,
                EventSeverity.HIGH,
                "Health entered the low-health band.",
                corroborated=True,
                current_health_percentage=health,
            )

    def _update_zero_health(
        self,
        signal,
        events,
    ):

        if signal.raw_percentage == 0.0:

            self._zero_count += 1

        else:

            self._zero_count = 0
            self._zero_emitted = False

        if (
            self._zero_count
            >= self.config.confirmation_samples
            and not self._zero_emitted
        ):

            self._emit(
                signal,
                events,
                EventType.ZERO_HEALTH_OBSERVED,
                EventSeverity.HIGH,
                "Consecutive zero-health measurements observed.",
                corroborated=True,
                current_health_percentage=0.0,
            )

            self._zero_emitted = True

    def _update_trend(
        self,
        signal,
        events,
        health,
    ):

        now = signal.monotonic_timestamp

        self._trend_history.append(
            (now, health)
        )

        window = self.config.trend_window_seconds

        cutoff = now - window

        while (
            self._trend_history
            and self._trend_history[0][0] < cutoff
        ):

            self._trend_history.popleft()

        if len(self._trend_history) < 2:

            return

        first_time, first_health = (
            self._trend_history[0]
        )

        duration = now - first_time

        if duration < window * 0.8:

            return

        change = health - first_health

        rate = change / duration

        event_type = None
        severity = None

        if (
            change <= -self.config.minimum_trend_change_pp
            and rate <= -self.config.decline_rate_threshold
        ):

            event_type = EventType.SUSTAINED_HEALTH_DECLINE

            severity = EventSeverity.HIGH

        elif (
            change >= self.config.minimum_trend_change_pp
            and rate >= self.config.recovery_rate_threshold
        ):

            event_type = EventType.SUSTAINED_HEALTH_RECOVERY

            severity = EventSeverity.MEDIUM

        if event_type is None:

            self._active_trend = None

            return

        if self._active_trend == event_type:

            return

        last_emitted = self._last_event_times.get(
            event_type
        )

        if (
            last_emitted is not None
            and now - last_emitted
            < self.config.trend_cooldown_seconds
        ):

            return

        self._emit(
            signal,
            events,
            event_type,
            severity,
            "Significant health trend observed "
            "over a continuous measurement window.",
            source="filtered_health_trend",
            corroborated=True,
            previous_health_percentage=first_health,
            current_health_percentage=health,
            change_pp=round(change, 4),
            rate_pp_per_second=round(rate, 4),
            window_seconds=round(duration, 4),
        )

        self._active_trend = event_type
