
from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass, asdict
from statistics import median

from aura.gameplay.temporal_tracker import (
    TemporalGameplayState,
)


@dataclass(frozen=True)
class SignalFilterConfig:

    slow_time_constant: float = 0.25
    fast_time_constant: float = 0.08

    deadband_pp: float = 0.5
    fast_response_threshold_pp: float = 2.0
    abrupt_change_threshold_pp: float = 5.0

    confirmation_tolerance_pp: float = 2.0

    noise_window_samples: int = 21
    minimum_noise_samples: int = 5

    maximum_measurement_gap: float = 0.5

    def __post_init__(self):

        values = (
            self.slow_time_constant,
            self.fast_time_constant,
            self.deadband_pp,
            self.fast_response_threshold_pp,
            self.abrupt_change_threshold_pp,
            self.confirmation_tolerance_pp,
            self.maximum_measurement_gap,
        )

        if any(
            not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError(
                "Filter parameters must be finite "
                "and positive."
            )

        if (
            self.fast_time_constant
            > self.slow_time_constant
        ):
            raise ValueError(
                "Fast time constant must not exceed "
                "slow time constant."
            )

        if not (
            self.deadband_pp
            < self.fast_response_threshold_pp
            < self.abrupt_change_threshold_pp
        ):
            raise ValueError(
                "Invalid response thresholds."
            )

        if not (
            2 <= self.minimum_noise_samples
            <= self.noise_window_samples
        ):
            raise ValueError(
                "Invalid noise estimation window."
            )


@dataclass(frozen=True)
class FilteredHealthSignal:

    frame_id: int
    monotonic_timestamp: float
    continuity_id: int

    available: bool

    raw_percentage: float | None = None
    filtered_percentage: float | None = None

    residual_pp: float | None = None
    raw_change_pp: float | None = None
    filtered_change_pp: float | None = None

    filtered_rate_pp_per_second: float | None = None

    estimated_noise_pp: float | None = None

    smoothing_alpha: float | None = None

    abrupt_change: bool = False
    abrupt_change_confirmed: bool = False

    mode: str = "unknown"

    reason: str | None = None

    def to_dict(self) -> dict:

        return asdict(self)


class HealthSignalFilter:
    """
    Causal, adaptive filtering of current health
    measurements.

    Raw evidence is never overwritten.

    Missing measurements remain unavailable.
    Large changes are preserved rather than
    automatically rejected as outliers.
    """

    def __init__(
        self,
        config: SignalFilterConfig | None = None,
    ):

        self.config = (
            config or SignalFilterConfig()
        )

        self.reset()

    def reset(self) -> None:

        self._filtered: float | None = None
        self._previous_raw: float | None = None

        self._last_valid_timestamp: float | None = None
        self._last_timestamp: float | None = None

        self._last_frame_id: int | None = None
        self._continuity_id: int | None = None

        self._pending_abrupt: float | None = None

        self._residuals = deque(
            maxlen=self.config.noise_window_samples
        )

    def _reset_signal(self) -> None:

        self._filtered = None
        self._previous_raw = None
        self._last_valid_timestamp = None

        self._pending_abrupt = None

        self._residuals.clear()

    def _estimate_noise(self) -> float | None:

        if (
            len(self._residuals)
            < self.config.minimum_noise_samples
        ):
            return None

        centre = median(self._residuals)

        mad = median(
            abs(value - centre)
            for value in self._residuals
        )

        # Robust scale estimate. This is a residual
        # variability proxy, not calibrated uncertainty.

        return 1.4826 * mad

    def update(
        self,
        state: TemporalGameplayState,
    ) -> FilteredHealthSignal:

        current = state.current

        now = current.monotonic_timestamp

        self._validate_order(
            current.frame_id,
            now,
        )

        self._last_timestamp = now
        self._last_frame_id = current.frame_id

        continuity_changed = (
            self._continuity_id is not None
            and state.continuity_id
            != self._continuity_id
        )

        self._continuity_id = (
            state.continuity_id
        )

        if continuity_changed:

            self._reset_signal()

        if not state.observation_available:

            self._reset_signal()

            return self._unavailable(
                state,
                "Gameplay observation unavailable.",
            )

        raw = state.current_health_percentage

        if raw is None:

            self._pending_abrupt = None

            if self._last_valid_timestamp is not None:

                gap = (
                    now - self._last_valid_timestamp
                )

                if (
                    gap
                    > self.config.maximum_measurement_gap
                ):
                    self._reset_signal()

            return self._unavailable(
                state,
                "Current health unavailable.",
            )

        if (
            not math.isfinite(raw)
            or not 0.0 <= raw <= 100.0
        ):

            return self._unavailable(
                state,
                "Invalid health measurement.",
            )

        if self._last_valid_timestamp is not None:

            gap = (
                now - self._last_valid_timestamp
            )

            if (
                gap
                > self.config.maximum_measurement_gap
            ):
                self._reset_signal()

        if (
            self._filtered is None
            or self._previous_raw is None
            or self._last_valid_timestamp is None
        ):

            return self._initialise(
                state,
                raw,
            )

        previous_filtered = self._filtered
        previous_raw = self._previous_raw

        dt = (
            now - self._last_valid_timestamp
        )

        raw_change = raw - previous_raw

        innovation = raw - previous_filtered

        noise = self._estimate_noise()

        # Noise can slightly increase the deadband,
        # but must not hide meaningful changes.

        effective_deadband = (
            self.config.deadband_pp
        )

        if noise is not None:

            effective_deadband = max(
                effective_deadband,
                min(
                    noise * 1.5,
                    self.config.fast_response_threshold_pp
                    * 0.75,
                ),
            )

        confirmed = False

        if self._pending_abrupt is not None:

            confirmed = (
                abs(raw - self._pending_abrupt)
                <= self.config.confirmation_tolerance_pp
            )

        self._pending_abrupt = None

        abrupt = (
            abs(raw_change)
            >= self.config.abrupt_change_threshold_pp
            or abs(innovation)
            >= self.config.abrupt_change_threshold_pp
        )

        if abrupt:

            # Preserve rapid changes immediately.
            # Confirmation is reported separately.

            filtered = raw
            alpha = 1.0

            mode = "abrupt"

            self._pending_abrupt = raw

            self._residuals.clear()

        elif raw in (0.0, 100.0):

            # Preserve exact observed endpoints.

            filtered = raw
            alpha = 1.0

            mode = "endpoint"

        elif abs(innovation) <= effective_deadband:

            filtered = previous_filtered
            alpha = 0.0

            mode = "deadband"

        else:

            if (
                abs(innovation)
                >= self.config.fast_response_threshold_pp
            ):

                tau = (
                    self.config.fast_time_constant
                )

                mode = "fast"

            else:

                tau = (
                    self.config.slow_time_constant
                )

                mode = "smooth"

            alpha = (
                1.0 - math.exp(-dt / tau)
            )

            filtered = (
                previous_filtered
                + alpha * innovation
            )

        filtered = min(
            100.0,
            max(0.0, filtered),
        )

        residual = raw - filtered

        filtered_change = (
            filtered - previous_filtered
        )

        if not abrupt:

            self._residuals.append(residual)

        self._filtered = filtered
        self._previous_raw = raw
        self._last_valid_timestamp = now

        rate = (
            filtered_change / dt
            if dt > 0
            else None
        )

        return FilteredHealthSignal(
            frame_id=current.frame_id,
            monotonic_timestamp=now,
            continuity_id=state.continuity_id,
            available=True,
            raw_percentage=round(raw, 4),
            filtered_percentage=round(
                filtered, 4
            ),
            residual_pp=round(residual, 4),
            raw_change_pp=round(
                raw_change, 4
            ),
            filtered_change_pp=round(
                filtered_change, 4
            ),
            filtered_rate_pp_per_second=(
                round(rate, 4)
                if rate is not None
                else None
            ),
            estimated_noise_pp=(
                round(noise, 4)
                if noise is not None
                else None
            ),
            smoothing_alpha=round(alpha, 4),
            abrupt_change=abrupt,
            abrupt_change_confirmed=confirmed,
            mode=mode,
        )

    def _initialise(
        self,
        state: TemporalGameplayState,
        raw: float,
    ) -> FilteredHealthSignal:

        self._filtered = raw
        self._previous_raw = raw

        self._last_valid_timestamp = (
            state.current.monotonic_timestamp
        )

        return FilteredHealthSignal(
            frame_id=state.current.frame_id,
            monotonic_timestamp=(
                state.current.monotonic_timestamp
            ),
            continuity_id=state.continuity_id,
            available=True,
            raw_percentage=raw,
            filtered_percentage=raw,
            residual_pp=0.0,
            smoothing_alpha=1.0,
            mode="initialised",
        )

    def _unavailable(
        self,
        state: TemporalGameplayState,
        reason: str,
    ) -> FilteredHealthSignal:

        return FilteredHealthSignal(
            frame_id=state.current.frame_id,
            monotonic_timestamp=(
                state.current.monotonic_timestamp
            ),
            continuity_id=state.continuity_id,
            available=False,
            mode="unavailable",
            reason=reason,
        )

    def _validate_order(
        self,
        frame_id: int,
        timestamp: float,
    ) -> None:

        if not math.isfinite(timestamp):

            raise ValueError(
                "Invalid monotonic timestamp."
            )

        if (
            self._last_timestamp is not None
            and timestamp < self._last_timestamp
        ):

            raise ValueError(
                "Out-of-order signal timestamp."
            )

        if (
            self._last_frame_id is not None
            and frame_id <= self._last_frame_id
        ):

            raise ValueError(
                "Frame IDs must increase."
            )
