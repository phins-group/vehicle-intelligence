"""Evidence-weighted temporal OCR aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from math import exp

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.config import VotingConfig
from vehicle_intelligence.domain import PlateCandidate, PlateObservation


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


class PlateCandidateAggregator:
    def __init__(self, config: VotingConfig, normalizer: VietnamPlateNormalizer) -> None:
        self._config = config
        self._normalizer = normalizer

    def aggregate(self, observations: Sequence[PlateObservation]) -> PlateCandidate | None:
        useful = [
            observation
            for observation in observations
            if observation.compact_text and observation.normalized_text
        ]
        if not useful:
            return None
        complete = [observation for observation in useful if not observation.partial]
        if complete:
            useful = complete
        clusters = self._clusters(useful)
        scored = [(self._cluster_score(cluster, len(useful)), cluster) for cluster in clusters]
        _, winner = max(scored, key=lambda item: (item[0], len(item[1])))
        consensus, character_agreement = self._character_consensus(winner)
        normalized = self._normalizer.normalize(consensus)
        if not normalized.valid or normalized.compact is None or normalized.normalized is None:
            best = max(winner, key=self._reliability)
            consensus = best.compact_text or ""
            normalized = self._normalizer.normalize(consensus)
        if not normalized.valid or normalized.compact is None or normalized.normalized is None:
            return None

        evidence_score = self._cluster_score(winner, len(useful), character_agreement)
        support = 1.0 - exp(-len(winner) / max(self._config.minimum_observations, 1))
        confidence = evidence_score + (1.0 - evidence_score) * 0.15 * support * character_agreement
        representative = min(
            winner,
            key=lambda item: (
                edit_distance(item.compact_text or "", normalized.compact or ""),
                -self._reliability(item),
            ),
        )
        corrections = normalized.corrections or representative.corrections
        return PlateCandidate(
            raw_text=representative.raw_text,
            normalized_text=normalized.normalized,
            compact_text=normalized.compact,
            confidence=min(max(confidence, 0.0), 0.999),
            observation_count=len(winner),
            corrections=corrections,
            partial=normalized.partial,
        )

    def _clusters(self, observations: list[PlateObservation]) -> list[list[PlateObservation]]:
        parents = list(range(len(observations)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(observations)):
            for right in range(left + 1, len(observations)):
                if (
                    edit_distance(
                        observations[left].compact_text or "",
                        observations[right].compact_text or "",
                    )
                    <= self._config.cluster_max_edit_distance
                ):
                    union(left, right)
        grouped: dict[int, list[PlateObservation]] = defaultdict(list)
        for index, observation in enumerate(observations):
            grouped[find(index)].append(observation)
        return list(grouped.values())

    def _cluster_score(
        self,
        cluster: Sequence[PlateObservation],
        total_count: int,
        character_consensus: float | None = None,
    ) -> float:
        config = self._config
        character_consensus = (
            self._character_consensus(cluster)[1]
            if character_consensus is None
            else character_consensus
        )
        components = (
            (len(cluster) / total_count, config.frequency_weight),
            (
                sum(item.ocr_confidence for item in cluster) / len(cluster),
                config.ocr_confidence_weight,
            ),
            (
                sum(item.quality_score for item in cluster) / len(cluster),
                config.quality_weight,
            ),
            (
                sum(item.detection_confidence for item in cluster) / len(cluster),
                config.detection_confidence_weight,
            ),
            (character_consensus, config.character_consensus_weight),
        )
        weight_total = sum(weight for _, weight in components)
        return sum(value * weight for value, weight in components) / weight_total

    def _character_consensus(self, cluster: Sequence[PlateObservation]) -> tuple[str, float]:
        lengths = Counter(len(item.compact_text or "") for item in cluster)
        target_length = max(lengths, key=lambda length: (lengths[length], length))
        aligned = [item for item in cluster if len(item.compact_text or "") == target_length]
        consensus: list[str] = []
        agreements: list[float] = []
        for position in range(target_length):
            votes: dict[str, float] = defaultdict(float)
            for item in aligned:
                text = item.compact_text or ""
                votes[text[position]] += self._reliability(item)
            character, winning_weight = max(votes.items(), key=lambda item: item[1])
            total_weight = sum(votes.values())
            consensus.append(character)
            agreements.append(winning_weight / total_weight if total_weight else 0.0)
        agreement = sum(agreements) / len(agreements) if agreements else 0.0
        return "".join(consensus), agreement

    @staticmethod
    def _reliability(observation: PlateObservation) -> float:
        return (
            observation.ocr_confidence
            * observation.quality_score
            * observation.detection_confidence
        ) ** (1 / 3)
