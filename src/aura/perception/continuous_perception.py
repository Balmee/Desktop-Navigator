
from __future__ import annotations

import time
import threading

from dataclasses import dataclass, asdict
from typing import Callable

from aura.perception.deadlock_observer import (
    DeadlockObserver,
)

from aura.perception.adaptive_health_estimator import (
    AdaptiveHealthEstimator,
)


@dataclass(frozen=True)
class PerceptionObservation:
    """
    Represents one timestamped gameplay observation.
    """

    frame_id: int

    timestamp: float
    monotonic_timestamp: float

    status: str

    frame_available: bool
    health_detected: bool

    health_fraction: float | None
    health_percentage: float | None

    measurement_quality: str

    capture_ms: float
    perception_ms: float
    total_processing_ms: float

    tracking_score: float | None = None
    tracking_scale: float | None = None

    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ContinuousPerceptionEngine:
    """
    Continuously observes Deadlock and produces
    structured gameplay measurements.

    This component does not perform language-model
    inference or generate coaching advice.
    """

    def __init__(
        self,
        target_hz: float = 10.0,
    ):

        if target_hz <= 0:
            raise ValueError(
                "Target frequency must be positive."
            )

        self.target_hz = target_hz
        self.interval = 1.0 / target_hz

        self.observer = DeadlockObserver()
        self.health_estimator = (
            AdaptiveHealthEstimator()
        )

        self._stop_event = threading.Event()
        self._running = False

        self.frame_id = 0

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_event.set()

    def observe_once(
        self,
    ) -> PerceptionObservation:

        self.frame_id += 1

        frame_id = self.frame_id

        start = time.perf_counter()
        timestamp = time.time()
        monotonic_timestamp = time.monotonic()

        capture_ms = 0.0
        perception_ms = 0.0

        def build_observation(
            status: str,
            frame_available: bool = False,
            health_detected: bool = False,
            health_fraction: float | None = None,
            health_percentage: float | None = None,
            quality: str = "unknown",
            tracking_score: float | None = None,
            tracking_scale: float | None = None,
            reason: str | None = None,
        ) -> PerceptionObservation:

            total_ms = (
                time.perf_counter() - start
            ) * 1000

            return PerceptionObservation(
                frame_id=frame_id,
                timestamp=timestamp,
                monotonic_timestamp=monotonic_timestamp,
                status=status,
                frame_available=frame_available,
                health_detected=health_detected,
                health_fraction=health_fraction,
                health_percentage=health_percentage,
                measurement_quality=quality,
                capture_ms=round(capture_ms, 3),
                perception_ms=round(perception_ms, 3),
                total_processing_ms=round(total_ms, 3),
                tracking_score=tracking_score,
                tracking_scale=tracking_scale,
                reason=reason,
            )

        # Stage 1: Capture the game window.

        capture_start = time.perf_counter()

        try:

            image = (
                self.observer.capture_game_window()
            )

        except Exception as error:

            capture_ms = (
                time.perf_counter() - capture_start
            ) * 1000

            return build_observation(
                status="capture_error",
                reason=str(error),
            )

        capture_ms = (
            time.perf_counter() - capture_start
        ) * 1000

        if image is None:

            return build_observation(
                status="unavailable",
                reason=(
                    "Deadlock is not available "
                    "for foreground observation."
                ),
            )

        # Stage 2: Extract health information.

        perception_start = time.perf_counter()

        try:

            result = self.health_estimator.estimate(
                image
            )

        except Exception as error:

            perception_ms = (
                time.perf_counter()
                - perception_start
            ) * 1000

            return build_observation(
                status="perception_error",
                frame_available=True,
                reason=str(error),
            )

        perception_ms = (
            time.perf_counter()
            - perception_start
        ) * 1000

        tracking = result.get("tracking") or {}

        if not result.get("detected", False):

            return build_observation(
                status="hud_unknown",
                frame_available=True,
                tracking_score=tracking.get("score"),
                tracking_scale=tracking.get("scale"),
                reason=result.get("reason"),
            )

        # Stage 3: Validate the measurement.

        fraction = result.get("health_fraction")

        if (
            fraction is None
            or not 0.0 <= fraction <= 1.0
        ):

            return build_observation(
                status="invalid_measurement",
                frame_available=True,
                reason="Invalid health fraction.",
            )

        return build_observation(
            status="observed",
            frame_available=True,
            health_detected=True,
            health_fraction=fraction,
            health_percentage=round(
                fraction * 100,
                2,
            ),
            quality=result.get(
                "quality",
                "provisional",
            ),
            tracking_score=tracking.get("score"),
            tracking_scale=tracking.get("scale"),
        )

    def run(
        self,
        on_observation: Callable[
            [PerceptionObservation],
            None,
        ],
        duration: float | None = None,
    ) -> None:

        if self._running:
            raise RuntimeError(
                "Perception engine is already running."
            )

        if duration is not None and duration <= 0:
            raise ValueError(
                "Duration must be positive."
            )

        self._stop_event.clear()
        self._running = True

        session_start = time.monotonic()

        try:

            while not self._stop_event.is_set():

                iteration_start = time.monotonic()

                elapsed = (
                    iteration_start - session_start
                )

                if (
                    duration is not None
                    and elapsed >= duration
                ):
                    break

                observation = self.observe_once()

                on_observation(observation)

                iteration_duration = (
                    time.monotonic()
                    - iteration_start
                )

                remaining = max(
                    0.0,
                    self.interval
                    - iteration_duration,
                )

                if duration is not None:

                    session_remaining = max(
                        0.0,
                        duration - (
                            time.monotonic()
                            - session_start
                        ),
                    )

                    remaining = min(
                        remaining,
                        session_remaining,
                    )

                self._stop_event.wait(remaining)

        finally:

            self._running = False
