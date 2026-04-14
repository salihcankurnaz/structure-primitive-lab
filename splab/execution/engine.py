"""Routed structure-aware execution engine."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.rewrite_executor import RewriteResult, normalize_expr
from splab.execution.routing import RoutingDecision, select_primitive
from splab.metrics.invariant_bridge import InvariantSignature, to_invariant_signature
from splab.metrics.structural_complexity import (
    StructuralMetrics,
    metrics_from_egraph,
    metrics_from_graph,
    metrics_from_hypergraph,
)
from splab.representations.egraph_hybrid import build_egraph_hybrid
from splab.representations.structure_graph import build_structure_graph
from splab.representations.typed_hypergraph import build_typed_hypergraph
from splab.tasks.symbolic_rewrite import Expr
from splab.tasks.task_case import TaskCase


@dataclass(frozen=True)
class EngineResult:
    rewrite: RewriteResult
    decision: RoutingDecision
    hyper_metrics: StructuralMetrics
    egraph_metrics: StructuralMetrics
    graph_metrics: StructuralMetrics
    invariant: InvariantSignature
    estimated_cost: float
    family: str | None
    metadata: dict[str, float | int | bool | str]


def run_engine(task: Expr | TaskCase) -> EngineResult:
    expr = task.expr if isinstance(task, TaskCase) else task
    metadata = task.metadata if isinstance(task, TaskCase) else {}
    family = task.family if isinstance(task, TaskCase) else None
    hyper = build_typed_hypergraph(expr)
    egraph = build_egraph_hybrid(expr)
    graph = build_structure_graph(expr)
    hyper_metrics = metrics_from_hypergraph(hyper)
    egraph_metrics = metrics_from_egraph(egraph)
    graph_metrics = metrics_from_graph(graph)
    decision = select_primitive(hyper_metrics, egraph_metrics, graph_metrics, family, metadata)
    if decision.primitive == "typed_hypergraph":
        invariant = to_invariant_signature("typed_hypergraph", hyper_metrics)
    elif decision.primitive == "egraph_hybrid":
        invariant = to_invariant_signature("egraph_hybrid", egraph_metrics)
    else:
        invariant = to_invariant_signature("structure_graph", graph_metrics)
    rewrite = normalize_expr(expr)
    return EngineResult(
        rewrite=rewrite,
        decision=decision,
        hyper_metrics=hyper_metrics,
        egraph_metrics=egraph_metrics,
        graph_metrics=graph_metrics,
        invariant=invariant,
        estimated_cost=decision.signal.estimated_cost,
        family=family,
        metadata=metadata,
    )
