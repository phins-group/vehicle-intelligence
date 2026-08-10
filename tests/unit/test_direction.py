from datetime import UTC, datetime, timedelta

from vehicle_intelligence.application.direction import DirectionEstimator, point_in_polygon
from vehicle_intelligence.domain import Direction, Point, TrajectoryPoint


def test_line_crossing_estimates_enter_and_reverse_exit() -> None:
    line = (Point(100, 50), Point(0, 50))
    estimator = DirectionEstimator(line, Direction.ENTER, "BOTH")
    start = datetime(2026, 8, 8, tzinfo=UTC)

    entering = [
        TrajectoryPoint(1, start, Point(50, 40)),
        TrajectoryPoint(2, start + timedelta(seconds=1), Point(50, 60)),
    ]
    exiting = list(reversed(entering))

    assert estimator.estimate(entering) is Direction.ENTER
    assert estimator.estimate(exiting) is Direction.EXIT


def test_camera_direction_is_fallback_without_line() -> None:
    assert DirectionEstimator(None, Direction.ENTER, "ENTRY").estimate(()) is Direction.ENTER
    assert DirectionEstimator(None, Direction.ENTER, "BOTH").estimate(()) is Direction.UNKNOWN


def test_point_in_polygon() -> None:
    polygon = (Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10))
    assert point_in_polygon(Point(5, 5), polygon)
    assert not point_in_polygon(Point(20, 5), polygon)
