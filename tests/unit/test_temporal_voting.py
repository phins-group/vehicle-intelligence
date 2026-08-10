from datetime import UTC, datetime, timedelta

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.voting import PlateCandidateAggregator, edit_distance
from vehicle_intelligence.config import VotingConfig
from vehicle_intelligence.domain import ModelMetadata, PlateObservation


def observation(text: str, confidence: float, index: int) -> PlateObservation:
    normalizer = VietnamPlateNormalizer()
    normalized = normalizer.normalize(text)
    model = ModelMetadata("test", "1")
    return PlateObservation(
        frame_id=index,
        timestamp=datetime(2026, 8, 8, tzinfo=UTC) + timedelta(milliseconds=index * 100),
        raw_text=text,
        normalized_text=normalized.normalized,
        compact_text=normalized.compact,
        ocr_confidence=confidence,
        detection_confidence=0.92,
        quality_score=0.88,
        corrections=normalized.corrections,
        plate_model=model,
        ocr_model=model,
    )


def test_weighted_temporal_vote_beats_noisy_observation() -> None:
    aggregator = PlateCandidateAggregator(VotingConfig(), VietnamPlateNormalizer())
    observations = [
        observation("51H12345", 0.94, 1),
        observation("51H12345", 0.91, 2),
        observation("51H1234S", 0.73, 3),
        observation("51H12345", 0.89, 4),
    ]

    result = aggregator.aggregate(observations)

    assert result is not None
    assert result.normalized_text == "51H-123.45"
    assert result.observation_count == 4
    assert result.confidence > 0.73


def test_edit_distance_handles_insert_delete_and_substitution() -> None:
    assert edit_distance("51H12345", "51H12345") == 0
    assert edit_distance("51H12345", "51H1234") == 1
    assert edit_distance("51H12345", "51H1234S") == 1


def test_partial_observation_is_retained_when_no_complete_plate_exists() -> None:
    normalizer = VietnamPlateNormalizer(allow_partial=True)
    normalized = normalizer.normalize("006.05")
    model = ModelMetadata("test", "1")
    partial = PlateObservation(
        frame_id=1,
        timestamp=datetime(2026, 8, 8, tzinfo=UTC),
        raw_text="006.05",
        normalized_text=normalized.normalized,
        compact_text=normalized.compact,
        ocr_confidence=0.99,
        detection_confidence=0.65,
        quality_score=0.72,
        corrections=(),
        plate_model=model,
        ocr_model=model,
        partial=normalized.partial,
    )

    result = PlateCandidateAggregator(VotingConfig(), normalizer).aggregate([partial])

    assert result is not None
    assert result.partial
    assert result.normalized_text == "006.05"


def test_complete_plate_is_preferred_over_partial_observations() -> None:
    normalizer = VietnamPlateNormalizer(allow_partial=True)
    model = ModelMetadata("test", "1")

    def make(text: str, index: int) -> PlateObservation:
        normalized = normalizer.normalize(text)
        return PlateObservation(
            frame_id=index,
            timestamp=datetime(2026, 8, 8, tzinfo=UTC) + timedelta(milliseconds=index),
            raw_text=text,
            normalized_text=normalized.normalized,
            compact_text=normalized.compact,
            ocr_confidence=0.95,
            detection_confidence=0.9,
            quality_score=0.9,
            corrections=normalized.corrections,
            plate_model=model,
            ocr_model=model,
            partial=normalized.partial,
        )

    result = PlateCandidateAggregator(VotingConfig(), normalizer).aggregate(
        [make("006.05", 1), make("006.05", 2), make("69F100605", 3)]
    )

    assert result is not None
    assert not result.partial
    assert result.normalized_text == "69F1-006.05"
