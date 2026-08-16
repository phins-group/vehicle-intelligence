"""Shared quality metric reduction without infrastructure dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_intelligence.domain import QualityMetrics


@dataclass(slots=True)
class QualityCounts:
    event_count: int = 0
    readable_plate_count: int = 0
    confirmed_count: int = 0
    needs_review_count: int = 0
    no_plate_count: int = 0
    unreadable_count: int = 0
    reviewed_count: int = 0
    corrected_count: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def add(self, other: QualityCounts) -> None:
        for field in (
            "event_count",
            "readable_plate_count",
            "confirmed_count",
            "needs_review_count",
            "no_plate_count",
            "unreadable_count",
            "reviewed_count",
            "corrected_count",
            "confidence_count",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))
        self.confidence_sum += other.confidence_sum

    def metrics(self) -> QualityMetrics:
        return QualityMetrics(
            event_count=self.event_count,
            readable_plate_count=self.readable_plate_count,
            confirmed_count=self.confirmed_count,
            needs_review_count=self.needs_review_count,
            no_plate_count=self.no_plate_count,
            unreadable_count=self.unreadable_count,
            reviewed_count=self.reviewed_count,
            corrected_count=self.corrected_count,
            ocr_success_rate=_ratio(self.readable_plate_count, self.event_count),
            unknown_plate_rate=_ratio(
                self.no_plate_count + self.unreadable_count,
                self.event_count,
            ),
            human_correction_rate=_ratio(self.corrected_count, self.reviewed_count),
            average_plate_confidence=(
                self.confidence_sum / self.confidence_count if self.confidence_count else None
            ),
        )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
