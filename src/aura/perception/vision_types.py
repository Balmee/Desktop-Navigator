
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image


class VisionResultStatus(str, Enum):

    SUCCESS = "success"
    ERROR = "error"
    STALE = "stale"


@dataclass(frozen=True)
class VisionRequest:

    request_id: int

    frame_id: int
    continuity_id: int
    generation: int

    captured_at: float
    submitted_at: float

    image: Image.Image

    reason: str

    context: dict[str, Any]


@dataclass(frozen=True)
class VisionResult:

    request_id: int

    frame_id: int
    continuity_id: int
    generation: int

    captured_at: float
    submitted_at: float

    started_at: float
    completed_at: float

    status: VisionResultStatus

    reason: str

    analysis: dict[str, Any] | None = None

    error: str | None = None

    @property
    def inference_ms(self) -> float:

        return (
            self.completed_at
            - self.started_at
        ) * 1000

    @property
    def queue_wait_ms(self) -> float:

        return (
            self.started_at
            - self.submitted_at
        ) * 1000

    @property
    def total_latency_ms(self) -> float:

        return (
            self.completed_at
            - self.submitted_at
        ) * 1000

    @property
    def evidence_age_ms(self) -> float:

        return (
            self.completed_at
            - self.captured_at
        ) * 1000

    def to_dict(self) -> dict:

        return {
            "request_id": self.request_id,
            "frame_id": self.frame_id,
            "continuity_id": self.continuity_id,
            "generation": self.generation,
            "captured_at": self.captured_at,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "reason": self.reason,
            "analysis": self.analysis,
            "error": self.error,
            "inference_ms": round(
                self.inference_ms, 4
            ),
            "queue_wait_ms": round(
                self.queue_wait_ms, 4
            ),
            "total_latency_ms": round(
                self.total_latency_ms, 4
            ),
            "evidence_age_ms": round(
                self.evidence_age_ms, 4
            ),
        }
