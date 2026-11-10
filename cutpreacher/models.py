from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in data["segments"]
        ]
        return cls(language=data["language"], duration=data["duration"], segments=segments)


@dataclass
class Clip:
    start: float
    end: float
    title: str
    hook: str
    reason: str
    description: str
    hashtags: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start
