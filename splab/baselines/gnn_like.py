"""Local propagation baseline."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.rewrite_executor import normalize_expr
from splab.representations.structure_graph import build_structure_graph
from splab.tasks.symbolic_rewrite import Expr


@dataclass(frozen=True)
class GNNBaselineResult:
    normalized: Expr
    estimated_cost: float
    passes: int


def run_gnn_like_baseline(expr: Expr, passes: int = 3) -> GNNBaselineResult:
    graph = build_structure_graph(expr)
    result = normalize_expr(expr)
    estimated_cost = float(graph.edge_count * passes)
    return GNNBaselineResult(
        normalized=result.expr,
        estimated_cost=estimated_cost,
        passes=passes,
    )
