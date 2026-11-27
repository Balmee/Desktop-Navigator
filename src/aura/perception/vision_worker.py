
from __future__ import annotations

import json
import math
import time

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Callable

from PIL import Image

from aura.perception.vision_types import (
    VisionRequest,
    VisionResult,
    VisionResultStatus,
)


VisionAnalyser = Callable[
    [VisionRequest],
    dict,
]


@dataclass(frozen=True)
class VisionWorkerConfig:

    maximum_request_age_seconds: float = 5.0

    maximum_result_age_seconds: float = 15.0

    maximum_buffered_results: int = 16

    def __post_init__(self):

        for value in (
            self.maximum_request_age_seconds,
            self.maximum_result_age_seconds,
        ):

            if not math.isfinite(value) or value <= 0:

                raise ValueError(
                    "Age limits must be finite "
                    "and positive."
                )

        if (
            self.maximum_result_age_seconds
            < self.maximum_request_age_seconds
        ):

            raise ValueError(
                "Result age limit must not be "
                "smaller than request age limit."
            )

        if self.maximum_buffered_results < 1:

            raise ValueError(
                "Result buffer must have "
                "positive capacity."
            )


class AsyncVisionWorker:
    """
    Single-consumer asynchronous vision worker.

    Guarantees:
    - One active inference operation.
    - At most one pending request.
    - Latest pending request replaces older requests.
    - Bounded completed-result buffer.
    - Generation-based context invalidation.
    - No screenshot persistence.
    """

    def __init__(
        self,
        analyser: VisionAnalyser,
        config: VisionWorkerConfig | None = None,
    ):

        self.analyser = analyser

        self.config = (
            config or VisionWorkerConfig()
        )

        self._condition = Condition()

        self._thread: Thread | None = None

        self._running = False

        self._pending: VisionRequest | None = None

        self._active: VisionRequest | None = None

        self._results = deque(
            maxlen=self.config.maximum_buffered_results
        )

        self._generation = 0

        self._continuity_id: int | None = None

        self._next_request_id = 0

        self._last_frame_id: int | None = None

        self._stats = {
            "submitted": 0,
            "started": 0,
            "succeeded": 0,
            "failed": 0,
            "replaced": 0,
            "expired_before_start": 0,
            "expired_after_inference": 0,
            "invalidated_pending": 0,
            "invalidated_results": 0,
            "invalidated_in_flight": 0,
            "result_buffer_overflows": 0,
            "expired_on_poll": 0,
        }

    def start(self) -> None:

        with self._condition:

            if self._running:

                return

            if (
                self._thread is not None
                and self._thread.is_alive()
            ):

                raise RuntimeError(
                    "Previous worker is still stopping."
                )

            self._running = True

            self._thread = Thread(
                target=self._run,
                name="AURA-VisionWorker",
                daemon=True,
            )

            self._thread.start()

    def submit(
        self,
        image: Image.Image,
        frame_id: int,
        continuity_id: int,
        captured_at: float,
        reason: str,
        context: dict | None = None,
    ) -> int:

        if not isinstance(image, Image.Image):

            raise TypeError(
                "Expected a Pillow image."
            )

        if not math.isfinite(captured_at):

            raise ValueError(
                "Invalid capture timestamp."
            )

        if captured_at > time.monotonic():

            raise ValueError(
                "Capture timestamp is in the future."
            )

        # Own an independent image and context snapshot.
        # The caller can safely continue using its frame.

        owned_image = image.copy()

        owned_context = deepcopy(
            context or {}
        )

        with self._condition:

            if not self._running:

                raise RuntimeError(
                    "Vision worker is not running."
                )

            if self._continuity_id is None:

                self._continuity_id = continuity_id

            elif continuity_id != self._continuity_id:

                self._invalidate_locked(
                    continuity_id
                )

            if (
                self._last_frame_id is not None
                and frame_id <= self._last_frame_id
            ):

                raise ValueError(
                    "Submitted frame IDs must increase "
                    "within the current generation."
                )

            self._last_frame_id = frame_id

            self._next_request_id += 1

            request = VisionRequest(
                request_id=self._next_request_id,
                frame_id=frame_id,
                continuity_id=continuity_id,
                generation=self._generation,
                captured_at=captured_at,
                submitted_at=time.monotonic(),
                image=owned_image,
                reason=reason,
                context=owned_context,
            )

            if self._pending is not None:

                self._stats["replaced"] += 1

            self._pending = request

            self._stats["submitted"] += 1

            self._condition.notify_all()

            return request.request_id

    def invalidate(
        self,
        continuity_id: int | None = None,
    ) -> None:

        with self._condition:

            self._invalidate_locked(
                continuity_id
            )

            self._condition.notify_all()

    def _invalidate_locked(
        self,
        continuity_id,
    ) -> None:

        self._generation += 1

        self._continuity_id = continuity_id

        self._last_frame_id = None

        if self._pending is not None:

            self._stats[
                "invalidated_pending"
            ] += 1

        self._pending = None

        self._stats[
            "invalidated_results"
        ] += len(self._results)

        self._results.clear()

    def poll_results(
        self,
    ) -> list[VisionResult]:

        with self._condition:

            now = time.monotonic()

            results = []

            while self._results:

                result = self._results.popleft()

                if (
                    result.generation
                    != self._generation
                ):

                    self._stats[
                        "invalidated_results"
                    ] += 1

                    continue

                age = now - result.captured_at

                if (
                    age
                    > self.config.maximum_result_age_seconds
                ):

                    self._stats[
                        "expired_on_poll"
                    ] += 1

                    continue

                results.append(result)

            return results

    def get_status(self) -> dict:

        with self._condition:

            return {
                "running": self._running,
                "busy": self._active is not None,
                "generation": self._generation,
                "continuity_id": self._continuity_id,
                "active_request_id": (
                    self._active.request_id
                    if self._active is not None
                    else None
                ),
                "pending_request_id": (
                    self._pending.request_id
                    if self._pending is not None
                    else None
                ),
                "pending_requests": int(
                    self._pending is not None
                ),
                "buffered_results": len(
                    self._results
                ),
                "statistics": dict(self._stats),
            }

    def wait_until_idle(
        self,
        timeout: float = 30.0,
    ) -> bool:

        with self._condition:

            return self._condition.wait_for(
                lambda: (
                    self._active is None
                    and self._pending is None
                ),
                timeout=timeout,
            )

    def stop(
        self,
        timeout: float = 5.0,
    ) -> bool:

        with self._condition:

            self._running = False

            self._invalidate_locked(None)

            self._condition.notify_all()

            thread = self._thread

        if thread is not None:

            thread.join(timeout=timeout)

            return not thread.is_alive()

        return True

    def _publish_locked(
        self,
        result: VisionResult,
    ) -> None:

        if (
            len(self._results)
            == self.config.maximum_buffered_results
        ):

            self._stats[
                "result_buffer_overflows"
            ] += 1

        self._results.append(result)

    def _run(self) -> None:

        while True:

            with self._condition:

                self._condition.wait_for(
                    lambda: (
                        not self._running
                        or self._pending is not None
                    )
                )

                if not self._running:

                    return

                request = self._pending

                self._pending = None

                now = time.monotonic()

                age = now - request.captured_at

                if (
                    age
                    > self.config.maximum_request_age_seconds
                ):

                    self._stats[
                        "expired_before_start"
                    ] += 1

                    self._condition.notify_all()

                    continue

                self._active = request

                self._stats["started"] += 1

                started_at = now

            # Never hold the condition lock during
            # expensive model inference.

            analysis = None

            error = None

            try:

                analysis = self.analyser(request)

                if not isinstance(analysis, dict):

                    raise TypeError(
                        "Vision analyser must return "
                        "a dictionary."
                    )

                # Validate serialisability before
                # accepting the model output.

                json.dumps(
                    analysis,
                    allow_nan=False,
                )

                analysis = deepcopy(analysis)

                status = VisionResultStatus.SUCCESS

            except Exception as exc:

                analysis = None

                status = VisionResultStatus.ERROR

                error = (
                    f"{type(exc).__name__}: {exc}"
                )

            completed_at = time.monotonic()

            with self._condition:

                invalidated = (
                    not self._running
                    or request.generation
                    != self._generation
                )

                expired = (
                    completed_at
                    - request.captured_at
                    > self.config.maximum_result_age_seconds
                )

                if invalidated:

                    self._stats[
                        "invalidated_in_flight"
                    ] += 1

                elif expired:

                    self._stats[
                        "expired_after_inference"
                    ] += 1

                else:

                    result = VisionResult(
                        request_id=request.request_id,
                        frame_id=request.frame_id,
                        continuity_id=request.continuity_id,
                        generation=request.generation,
                        captured_at=request.captured_at,
                        submitted_at=request.submitted_at,
                        started_at=started_at,
                        completed_at=completed_at,
                        status=status,
                        reason=request.reason,
                        analysis=analysis,
                        error=error,
                    )

                    self._publish_locked(result)

                    if status == VisionResultStatus.SUCCESS:

                        self._stats["succeeded"] += 1

                    else:

                        self._stats["failed"] += 1

                self._active = None

                self._condition.notify_all()
