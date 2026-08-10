import pytest

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("51H12345", "51H-123.45"),
        ("51H 12345", "51H-123.45"),
        ("51H-12345", "51H-123.45"),
        ("51H123.45", "51H-123.45"),
        ("59X388217", "59X3-882.17"),
        ("30LD12345", "30LD-123.45"),
        ("29A1234", "29A-1234"),
    ],
)
def test_normalizes_common_vietnamese_formats(raw: str, expected: str) -> None:
    result = VietnamPlateNormalizer().normalize(raw)

    assert result.valid
    assert result.normalized == expected


def test_applies_character_confusion_only_in_digit_position() -> None:
    result = VietnamPlateNormalizer().normalize("51HI2345")

    assert result.normalized == "51H-123.45"
    assert len(result.corrections) == 1
    correction = result.corrections[0]
    assert (
        correction.position,
        correction.from_character,
        correction.to_character,
    ) == (3, "I", "1")


@pytest.mark.parametrize("raw", ["HELLO", "123", "51-@@@@", "51H123456789"])
def test_rejects_invalid_plate_without_fabricating_canonical_text(raw: str) -> None:
    result = VietnamPlateNormalizer().normalize(raw)

    assert not result.valid
    assert result.normalized is None
    assert result.compact is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("006.05", "006.05"),
        ("393.45", "393.45"),
        ("03NI39345", "03NI39345"),
    ],
)
def test_lenient_mode_keeps_partial_ocr_for_review(raw: str, expected: str) -> None:
    result = VietnamPlateNormalizer(allow_partial=True).normalize(raw)

    assert result.valid
    assert result.partial
    assert result.normalized == expected
    assert result.compact == expected.replace(".", "")


@pytest.mark.parametrize("raw", ["HELLO", "12", "1234567890123"])
def test_lenient_mode_still_rejects_obvious_ocr_noise(raw: str) -> None:
    result = VietnamPlateNormalizer(allow_partial=True).normalize(raw)

    assert not result.valid
    assert not result.partial
