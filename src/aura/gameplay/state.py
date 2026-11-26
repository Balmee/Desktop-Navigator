
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Generic, TypeVar


T = TypeVar("T")


class EvidenceStatus(str, Enum):

    CURRENT = "current"
    HISTORICAL = "historical"
    STALE = "stale"
    UNKNOWN = "unknown"


class EvidenceSource(str, Enum):

    COMPUTER_VISION = "computer_vision"
    OCR = "ocr"
    VISION_MODEL = "vision_model"
    DERIVED = "derived"
    USER = "user"
    UNKNOWN = "unknown"


class ObservationStatus(str, Enum):

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    HUD_UNKNOWN = "hud_unknown"
    CAPTURE_ERROR = "capture_error"
    PERCEPTION_ERROR = "perception_error"
    INVALID_MEASUREMENT = "invalid_measurement"


@dataclass(frozen=True)
class Evidence(Generic[T]):
    """
    A value together with its provenance and
    observation metadata.
    """

    value: T | None = None

    status: EvidenceStatus = (
        EvidenceStatus.UNKNOWN
    )

    source: EvidenceSource = (
        EvidenceSource.UNKNOWN
    )

    timestamp: float | None = None
    monotonic_timestamp: float | None = None

    quality: str = "unknown"

    confidence: float | None = None

    reason: str | None = None

    @property
    def is_current(self) -> bool:

        return (
            self.status == EvidenceStatus.CURRENT
            and self.value is not None
        )

    @classmethod
    def unknown(
        cls,
        reason: str | None = None,
    ) -> Evidence[T]:

        return cls(reason=reason)


@dataclass(frozen=True)
class PlayerState:

    hero: Evidence[str] = field(
        default_factory=Evidence
    )

    health_fraction: Evidence[float] = field(
        default_factory=Evidence
    )

    current_health: Evidence[int] = field(
        default_factory=Evidence
    )

    maximum_health: Evidence[int] = field(
        default_factory=Evidence
    )

    location: Evidence[str] = field(
        default_factory=Evidence
    )

    alive: Evidence[bool] = field(
        default_factory=Evidence
    )

    status_effects: Evidence[tuple[str, ...]] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class ResourceState:

    currency: Evidence[int] = field(
        default_factory=Evidence
    )

    ammunition: Evidence[int] = field(
        default_factory=Evidence
    )

    stamina: Evidence[float] = field(
        default_factory=Evidence
    )

    item_value: Evidence[int] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class AbilityState:

    slot: int

    name: Evidence[str] = field(
        default_factory=Evidence
    )

    available: Evidence[bool] = field(
        default_factory=Evidence
    )

    cooldown_seconds: Evidence[float] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class EntityState:

    name: Evidence[str] = field(
        default_factory=Evidence
    )

    entity_type: Evidence[str] = field(
        default_factory=Evidence
    )

    team: Evidence[str] = field(
        default_factory=Evidence
    )

    screen_position: Evidence[str] = field(
        default_factory=Evidence
    )

    health_fraction: Evidence[float] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class EnvironmentState:

    description: Evidence[str] = field(
        default_factory=Evidence
    )

    minimap_visible: Evidence[bool] = field(
        default_factory=Evidence
    )

    game_mode: Evidence[str] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class CombatState:

    active_engagement: Evidence[bool] = field(
        default_factory=Evidence
    )

    incoming_damage: Evidence[bool] = field(
        default_factory=Evidence
    )

    visible_threats: Evidence[tuple[str, ...]] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class ObjectiveState:

    name: Evidence[str] = field(
        default_factory=Evidence
    )

    status: Evidence[str] = field(
        default_factory=Evidence
    )

    location: Evidence[str] = field(
        default_factory=Evidence
    )


@dataclass(frozen=True)
class GameplayState:
    """
    Immutable snapshot of currently available
    gameplay evidence.
    """

    frame_id: int

    timestamp: float
    monotonic_timestamp: float

    observation_status: ObservationStatus

    frame_available: bool

    player: PlayerState = field(
        default_factory=PlayerState
    )

    resources: ResourceState = field(
        default_factory=ResourceState
    )

    abilities: tuple[AbilityState, ...] = field(
        default_factory=lambda: tuple(
            AbilityState(slot=i)
            for i in range(1, 5)
        )
    )

    entities: Evidence[
        tuple[EntityState, ...]
    ] = field(default_factory=Evidence)

    environment: EnvironmentState = field(
        default_factory=EnvironmentState
    )

    combat: CombatState = field(
        default_factory=CombatState
    )

    objectives: Evidence[
        tuple[ObjectiveState, ...]
    ] = field(default_factory=Evidence)

    reason: str | None = None

    @property
    def health_available(self) -> bool:

        return self.player.health_fraction.is_current

    @property
    def health_percentage(self) -> float | None:

        if not self.health_available:
            return None

        fraction = self.player.health_fraction.value

        if fraction is None:
            return None

        return round(fraction * 100, 2)

    def to_dict(self) -> dict:

        return asdict(self)
