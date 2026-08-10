"""Vietnamese plate normalization with position-aware confusion handling."""

from __future__ import annotations

import re
from dataclasses import dataclass

from vehicle_intelligence.domain import CharacterCorrection, PlateNormalization

_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
_DIGIT_TO_LETTER = {value: key for key, value in _LETTER_TO_DIGIT.items()}

# Two-character series used by special-purpose Vietnamese registrations. A
# letter+digit motorcycle series is accepted independently of this set.
_KNOWN_TWO_LETTER_SERIES = {
    "CD",
    "CV",
    "DA",
    "HC",
    "KT",
    "LA",
    "LB",
    "LD",
    "MA",
    "MK",
    "NG",
    "NN",
    "QT",
    "RM",
    "TĐ",
}


@dataclass(frozen=True, slots=True)
class _StructuralCandidate:
    compact: str
    series_length: int
    corrections: tuple[CharacterCorrection, ...]

    @property
    def serial_length(self) -> int:
        return len(self.compact) - 2 - self.series_length


class VietnamPlateNormalizer:
    """Normalize common civilian and special Vietnamese plate layouts."""

    correction_confidence = 0.91

    def __init__(
        self,
        *,
        allow_partial: bool = False,
        partial_min_characters: int = 4,
        partial_max_characters: int = 12,
    ) -> None:
        if partial_min_characters < 1:
            raise ValueError("partial plate minimum must be positive")
        if partial_max_characters < partial_min_characters:
            raise ValueError("partial plate maximum cannot be below its minimum")
        self._allow_partial = allow_partial
        self._partial_min_characters = partial_min_characters
        self._partial_max_characters = partial_max_characters

    def normalize(self, raw: str) -> PlateNormalization:
        cleaned = _NON_ALNUM.sub("", raw.upper())
        candidates = self._structural_candidates(cleaned)
        if not candidates:
            partial = self._partial_normalization(raw, cleaned)
            if partial is not None:
                return partial
            return PlateNormalization(
                raw=raw,
                cleaned=cleaned,
                compact=None,
                normalized=None,
                valid=False,
            )

        # Prefer fewer corrections, then five-digit serials, then the conventional
        # single-letter car series. This resolves 51HI2345 as 51H12345 rather than
        # inventing the unknown two-letter series HI.
        winner = min(
            candidates,
            key=lambda item: (
                len(item.corrections),
                0 if item.serial_length == 5 else 1,
                item.series_length,
            ),
        )
        prefix_length = 2 + winner.series_length
        prefix = winner.compact[:prefix_length]
        serial = winner.compact[prefix_length:]
        formatted_serial = f"{serial[:3]}.{serial[3:]}" if len(serial) == 5 else serial
        return PlateNormalization(
            raw=raw,
            cleaned=cleaned,
            compact=winner.compact,
            normalized=f"{prefix}-{formatted_serial}",
            valid=True,
            corrections=winner.corrections,
        )

    def _partial_normalization(
        self,
        raw: str,
        cleaned: str,
    ) -> PlateNormalization | None:
        if not self._allow_partial:
            return None
        if not self._partial_min_characters <= len(cleaned) <= self._partial_max_characters:
            return None
        # Requiring at least one digit is a noise guard, not a Vietnamese plate
        # structure rule. It prevents ordinary OCR words from becoming identities.
        if not any(character.isdigit() for character in cleaned):
            return None
        normalized = (
            f"{cleaned[:3]}.{cleaned[3:]}" if len(cleaned) == 5 and cleaned.isdigit() else cleaned
        )
        return PlateNormalization(
            raw=raw,
            cleaned=cleaned,
            compact=cleaned,
            normalized=normalized,
            valid=True,
            partial=True,
        )

    def _structural_candidates(self, cleaned: str) -> list[_StructuralCandidate]:
        if not 7 <= len(cleaned) <= 9:
            return []
        candidates: list[_StructuralCandidate] = []
        for series_length in (1, 2):
            serial_length = len(cleaned) - 2 - series_length
            if serial_length not in (4, 5):
                continue
            corrected = list(cleaned)
            corrections: list[CharacterCorrection] = []
            if not self._coerce_positions(corrected, range(0, 2), True, corrections):
                continue
            if not self._coerce_positions(corrected, (2,), False, corrections):
                continue
            if series_length == 2:
                second = corrected[3]
                if second.isalpha():
                    pair = "".join(corrected[2:4])
                    if pair not in _KNOWN_TWO_LETTER_SERIES:
                        continue
                elif not second.isdigit():
                    continue
            serial_positions = range(2 + series_length, len(corrected))
            if not self._coerce_positions(corrected, serial_positions, True, corrections):
                continue
            compact = "".join(corrected)
            candidates.append(_StructuralCandidate(compact, series_length, tuple(corrections)))
        return candidates

    def _coerce_positions(
        self,
        characters: list[str],
        positions: object,
        expect_digit: bool,
        corrections: list[CharacterCorrection],
    ) -> bool:
        for position in positions:  # type: ignore[union-attr]
            character = characters[position]
            if expect_digit and character.isdigit():
                continue
            if not expect_digit and character.isalpha():
                continue
            replacement = (
                _LETTER_TO_DIGIT.get(character) if expect_digit else _DIGIT_TO_LETTER.get(character)
            )
            if replacement is None:
                return False
            characters[position] = replacement
            corrections.append(
                CharacterCorrection(
                    position=position,
                    from_character=character,
                    to_character=replacement,
                    confidence=self.correction_confidence,
                )
            )
        return True
