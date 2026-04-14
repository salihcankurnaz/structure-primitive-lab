"""Heuristic exact baseline."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.rewrite_executor import normalize_expr
from splab.tasks.symbolic_rewrite import Expr, expr_size


@dataclass(frozen=True)
class BaselineResult:
    normalized: Expr
    estimated_cost: float
    steps: tuple[str, ...]


def run_heuristic_baseline(expr: Expr) -> BaselineResult:
    result = normalize_expr(expr)
    return BaselineResult(
        normalized=result.expr,
        estimated_cost=float(expr_size(expr)),
        steps=result.steps,
    )
