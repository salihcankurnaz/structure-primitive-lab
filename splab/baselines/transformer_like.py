"""Dense global-scoring baseline."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.rewrite_executor import normalize_expr
from splab.tasks.symbolic_rewrite import Expr, expr_size


@dataclass(frozen=True)
class TransformerBaselineResult:
    normalized: Expr
    estimated_cost: float
    token_count: int


def run_transformer_like_baseline(expr: Expr) -> TransformerBaselineResult:
    token_count = expr_size(expr)
    result = normalize_expr(expr)
    return TransformerBaselineResult(
        normalized=result.expr,
        estimated_cost=float(token_count * token_count),
        token_count=token_count,
    )
