"""Placeholders for future GPU lowering."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.engine import EngineResult


@dataclass(frozen=True)
class LoweringPlan:
    primitive: str
    backend: str
    note: str


def plan_gpu_lowering(result: EngineResult) -> LoweringPlan:
    if result.decision.primitive == "egraph_hybrid":
        return LoweringPlan("egraph_hybrid", "gather-scatter", "equivalence buckets to segmented ops")
    if result.decision.primitive == "typed_hypergraph":
        return LoweringPlan("typed_hypergraph", "sparse_tensor", "higher-arity relations to sparse blocks")
    return LoweringPlan("structure_graph", "message_passing", "binary edges to sparse propagation")
