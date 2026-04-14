"""Shared structural metrics across representations."""

from __future__ import annotations

from dataclasses import dataclass

from splab.representations.egraph_hybrid import EGraphHybrid
from splab.representations.structure_graph import StructureGraph
from splab.representations.typed_hypergraph import TypedHypergraph


@dataclass(frozen=True)
class StructuralMetrics:
    size: int
    relation_count: int
    max_arity: int
    equivalence_density: int
    score: float


def metrics_from_hypergraph(rep: TypedHypergraph) -> StructuralMetrics:
    score = rep.node_count + 1.5 * len(rep.edges) + 2.0 * rep.max_arity
    return StructuralMetrics(
        size=rep.node_count,
        relation_count=len(rep.edges),
        max_arity=rep.max_arity,
        equivalence_density=0,
        score=score,
    )


def metrics_from_egraph(rep: EGraphHybrid) -> StructuralMetrics:
    score = rep.class_count + 0.5 * rep.equivalence_density
    return StructuralMetrics(
        size=rep.class_count,
        relation_count=rep.class_count,
        max_arity=1,
        equivalence_density=rep.equivalence_density,
        score=score,
    )


def metrics_from_graph(rep: StructureGraph) -> StructuralMetrics:
    score = rep.node_count + 0.75 * rep.edge_count
    return StructuralMetrics(
        size=rep.node_count,
        relation_count=rep.edge_count,
        max_arity=2,
        equivalence_density=0,
        score=score,
    )
