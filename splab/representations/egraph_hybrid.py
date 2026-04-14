"""Simple e-graph-like hybrid representation."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.rewrite_executor import normalize_expr
from splab.tasks.symbolic_rewrite import Expr, pretty_expr


@dataclass(frozen=True)
class EClass:
    class_id: int
    members: tuple[str, ...]


@dataclass(frozen=True)
class EGraphHybrid:
    classes: tuple[EClass, ...]
    root_key: str

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def equivalence_density(self) -> int:
        return sum(len(eclass.members) for eclass in self.classes)


def build_egraph_hybrid(expr: Expr) -> EGraphHybrid:
    normal = normalize_expr(expr).expr
    original = pretty_expr(expr)
    canonical = pretty_expr(normal)
    members = tuple(sorted({original, canonical}))
    return EGraphHybrid(
        classes=(EClass(class_id=0, members=members),),
        root_key=canonical,
    )
