from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"bounding box must have positive area: {self}")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clip(self, image_width: int, image_height: int) -> BoundingBox | None:
        x1 = min(max(self.x1, 0), image_width)
        y1 = min(max(self.y1, 0), image_height)
        x2 = min(max(self.x2, 0), image_width)
        y2 = min(max(self.y2, 0), image_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    def translate(self, dx: int, dy: int) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
