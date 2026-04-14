"""Bridge representation summaries to comparable invariant signatures."""

from __future__ import annotations

from dataclasses import dataclass

from splab.metrics.structural_complexity import StructuralMetrics


@dataclass(frozen=True)
class InvariantSignature:
    primitive: str
    size: int
    relation_count: int
    score: float


def to_invariant_signature(primitive: str, metrics: StructuralMetrics) -> InvariantSignature:
    return InvariantSignature(
        primitive=primitive,
        size=metrics.size,
        relation_count=metrics.relation_count,
        score=round(metrics.score, 3),
    )
