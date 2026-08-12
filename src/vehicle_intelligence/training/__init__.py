"""Offline detector dataset, training, evaluation, and promotion tooling.

Nothing in the online camera or vision pipeline imports this package.  The
training boundary may consume canonical detector configuration and artifacts,
but production inference stays independent of training infrastructure.
"""

from vehicle_intelligence.training.domain import DatasetSplit, DetectorRole

__all__ = ["DatasetSplit", "DetectorRole"]
