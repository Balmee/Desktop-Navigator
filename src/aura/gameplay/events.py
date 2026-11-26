
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class EventType(str, Enum):

    OBSERVATION_LOST = "observation_lost"
    OBSERVATION_RESTORED = "observation_restored"

    HEALTH_SIGNAL_LOST = "health_signal_lost"
    HEALTH_SIGNAL_RESTORED = "health_signal_restored"

    RAPID_HEALTH_LOSS = "rapid_health_loss"
    RAPID_HEALTH_GAIN = "rapid_health_gain"

    SUSTAINED_HEALTH_DECLINE = "sustained_health_decline"
    SUSTAINED_HEALTH_RECOVERY = "sustained_health_recovery"

    LOW_HEALTH_ENTERED = "low_health_entered"
    LOW_HEALTH_EXITED = "low_health_exited"

    CRITICAL_HEALTH_ENTERED = "critical_health_entered"
    CRITICAL_HEALTH_EXITED = "critical_health_exited"

    ZERO_HEALTH_OBSERVED = "zero_health_observed"


class EventSeverity(str, Enum):

    SYSTEM = "system"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthBand(str, Enum):

    UNKNOWN = "unknown"
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"


@dataclass(frozen=True)
class GameplayEvent:

    event_id: int

    event_type: EventType
    severity: EventSeverity

    frame_id: int
    monotonic_timestamp: float
    continuity_id: int

    source: str

    description: str

    corroborated: bool = False

    previous_health_percentage: float | None = None
    current_health_percentage: float | None = None

    change_pp: float | None = None
    rate_pp_per_second: float | None = None

    window_seconds: float | None = None

    originating_frame_id: int | None = None

    def to_dict(self) -> dict:

        return asdict(self)


@dataclass(frozen=True)
class EventDetectionResult:

    frame_id: int
    monotonic_timestamp: float
    continuity_id: int

    observation_available: bool
    health_available: bool

    health_band: HealthBand

    pending_abrupt_change: bool

    events: tuple[GameplayEvent, ...]

    def to_dict(self) -> dict:

        return asdict(self)
