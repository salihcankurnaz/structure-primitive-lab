"""Simple sparse message passing signals over structure representations."""

from __future__ import annotations

from dataclasses import dataclass

from splab.metrics.structural_complexity import StructuralMetrics


@dataclass(frozen=True)
class MessagePassingSignal:
    primitive: str
    predicted_gain: float
    estimated_cost: float


def estimate_signal(primitive: str, metrics: StructuralMetrics) -> MessagePassingSignal:
    if primitive == "typed_hypergraph":
        gain = 1.5 + 0.3 * metrics.max_arity
        cost = max(1.0, 0.7 * metrics.score)
    elif primitive == "egraph_hybrid":
        gain = 1.0 + 0.2 * metrics.equivalence_density
        cost = max(1.0, 0.5 * metrics.score)
    else:
        gain = 1.0 + 0.1 * metrics.relation_count
        cost = max(1.0, 0.9 * metrics.score)
    return MessagePassingSignal(
        primitive=primitive,
        predicted_gain=round(gain, 3),
        estimated_cost=round(cost, 3),
    )
