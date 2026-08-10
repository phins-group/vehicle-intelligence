"""Trajectory and line-crossing domain service."""

from __future__ import annotations

from collections.abc import Sequence

from vehicle_intelligence.domain import Direction, Point, TrajectoryPoint


def signed_side(point: Point, line_start: Point, line_end: Point) -> float:
    return (line_end.x - line_start.x) * (point.y - line_start.y) - (line_end.y - line_start.y) * (
        point.x - line_start.x
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        intersects = (current.y > point.y) != (previous.y > point.y) and point.x < (
            (previous.x - current.x) * (point.y - current.y) / ((previous.y - current.y) or 1e-12)
            + current.x
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


class DirectionEstimator:
    def __init__(
        self,
        line: tuple[Point, Point] | None,
        positive_to_negative: Direction,
        camera_direction: str,
    ) -> None:
        self._line = line
        self._positive_to_negative = positive_to_negative
        self._camera_direction = camera_direction

    def estimate(self, trajectory: Sequence[TrajectoryPoint]) -> Direction:
        if self._line is not None and len(trajectory) >= 2:
            start, end = self._line
            previous_side = signed_side(trajectory[0].center, start, end)
            for point in trajectory[1:]:
                current_side = signed_side(point.center, start, end)
                if previous_side > 0 >= current_side:
                    return self._positive_to_negative
                if previous_side < 0 <= current_side:
                    return self._opposite(self._positive_to_negative)
                if current_side != 0:
                    previous_side = current_side
        if self._camera_direction == "ENTRY":
            return Direction.ENTER
        if self._camera_direction == "EXIT":
            return Direction.EXIT
        return Direction.UNKNOWN

    @staticmethod
    def _opposite(direction: Direction) -> Direction:
        return Direction.EXIT if direction is Direction.ENTER else Direction.ENTER
