"""Internal dataclasses (SPEC section 6). Provider shapes never leak past the adapters."""
from dataclasses import asdict, dataclass, field
from typing import Literal

Granularity = Literal["word", "segment", "none"]


@dataclass
class Unit:
    text: str
    t_start: float | None = None
    t_end: float | None = None
    confidence: float | None = None


@dataclass
class Transcript:
    text: str
    granularity: Granularity
    units: list[Unit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.granularity == "none":
            return
        for unit in self.units:
            if unit.t_start is None or unit.t_end is None:
                raise ValueError(f"granularity {self.granularity!r} requires timings on every unit")
            if unit.t_end < unit.t_start:
                raise ValueError("unit t_end before t_start")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(data["text"], data["granularity"], [Unit(**u) for u in data.get("units", [])])


@dataclass
class FrameHit:
    frame_idx: int
    t: float
    box_xyxy: tuple[float, float, float, float]
    mask_path: str | None = None
    score: float | None = None


@dataclass
class Track:
    track_id: str
    label: str
    frames: list[FrameHit] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        hits = [FrameHit(h["frame_idx"], h["t"], tuple(h["box_xyxy"]), h.get("mask_path"), h.get("score"))
                for h in data.get("frames", [])]
        return cls(data["track_id"], data.get("label", ""), hits)


@dataclass
class Crop:
    track_id: str
    image_path: str


@dataclass
class ProductGuess:
    track_id: str
    brand: str | None
    name: str | None
    size: str | None
    confidence: float | None


@dataclass
class PriceMention:
    text: str
    amount_pesewas: int
    t_start: float | None
    t_end: float | None
    referent: str | None = None
    source: Literal["model", "regex", "fixture"] = "model"

    def __post_init__(self) -> None:
        if not isinstance(self.amount_pesewas, int) or self.amount_pesewas <= 0:
            raise ValueError("amount_pesewas must be a positive integer")


@dataclass
class Candidate:
    track_id: str
    score: float
    visibility: float = 0.0
    centrality: float = 0.0
    size: float = 0.0
    name_bonus: float = 0.0


@dataclass
class RankedCandidate:
    track_id: str
    rank: int
    confidence: float | None


@dataclass
class Msg:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class AgentReply:
    text: str | None
    tool_calls: list[dict] = field(default_factory=list)
