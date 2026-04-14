"""Primitive routing based on structure metrics."""

from __future__ import annotations

from dataclasses import dataclass

from splab.execution.calibration import PRIMITIVES, load_calibration, metadata_signature
from splab.execution.sparse_message_passing import MessagePassingSignal, estimate_signal
from splab.metrics.structural_complexity import StructuralMetrics


@dataclass(frozen=True)
class RoutingDecision:
    primitive: str
    reason: str
    signal: MessagePassingSignal
    candidate_costs: dict[str, float]


def select_primitive(
    hyper_metrics: StructuralMetrics,
    egraph_metrics: StructuralMetrics,
    graph_metrics: StructuralMetrics,
    family: str | None = None,
    metadata: dict[str, float | int | bool | str] | None = None,
) -> RoutingDecision:
    metadata = metadata or {}
    hyper_signal = estimate_signal("typed_hypergraph", hyper_metrics)
    egraph_signal = estimate_signal("egraph_hybrid", egraph_metrics)
    graph_signal = estimate_signal("structure_graph", graph_metrics)
    calibration = load_calibration()
    candidate_costs = {
        "typed_hypergraph": hyper_signal.estimated_cost,
        "egraph_hybrid": egraph_signal.estimated_cost,
        "structure_graph": graph_signal.estimated_cost,
    }
    if calibration:
        candidate_costs = {
            primitive: calibration.adjusted_cost(primitive, family, metadata, base_cost)
            for primitive, base_cost in candidate_costs.items()
        }
        family_oracle = calibration.family_oracles.get(family or "")
        family_confidence = calibration.family_oracle_confidence.get(family or "", 0.0)
        if family_oracle and family_confidence >= 0.95:
            candidate_costs[family_oracle] *= 0.72
            reasons = [f"oracle_family_bias:{family_oracle}"]
        else:
            reasons = []
        signature = metadata_signature(family, metadata)
        oracle_primitive = calibration.signature_oracles.get(signature)
        oracle_confidence = calibration.signature_oracle_confidence.get(signature, 0.0)
        if oracle_primitive and oracle_confidence >= 0.75:
            candidate_costs[oracle_primitive] *= 0.82
            reasons.append(f"oracle_signature_bias:{oracle_primitive}")
    else:
        reasons = []
    if bool(metadata.get("hypergraph_preferred")):
        candidate_costs["typed_hypergraph"] *= 0.88
        reasons.append("hypergraph_bias")
    if bool(metadata.get("hyperedge_native")):
        candidate_costs["typed_hypergraph"] *= 0.55
        candidate_costs["egraph_hybrid"] *= 1.35
        reasons.append("hyperedge_native_bias")
    if bool(metadata.get("constraint_native")):
        candidate_costs["typed_hypergraph"] *= 0.45
        candidate_costs["egraph_hybrid"] *= 1.6
        candidate_costs["structure_graph"] *= 1.15
        reasons.append("constraint_native_bias")
    if float(metadata.get("relation_arity", 0)) > 2 or hyper_metrics.max_arity > 2:
        candidate_costs["typed_hypergraph"] *= 0.9
        reasons.append("higher_arity_bias")
    if bool(metadata.get("graph_preferred")):
        candidate_costs["structure_graph"] *= 0.9
        reasons.append("graph_bias")
    if bool(metadata.get("graph_native")):
        candidate_costs["structure_graph"] *= 0.5
        candidate_costs["egraph_hybrid"] *= 1.45
        candidate_costs["typed_hypergraph"] *= 1.1
        reasons.append("graph_native_bias")
    if float(metadata.get("aggregation_bias", 0)) >= 4:
        candidate_costs["typed_hypergraph"] *= 0.6
        candidate_costs["structure_graph"] *= 0.95
        reasons.append("aggregation_bias_high")
    elif float(metadata.get("aggregation_bias", 0)) >= 2:
        candidate_costs["typed_hypergraph"] *= 0.88
        reasons.append("aggregation_bias_mid")
    if bool(metadata.get("flow_heavy")):
        candidate_costs["structure_graph"] *= 0.92
        reasons.append("flow_bias")
    if bool(metadata.get("equivalence_heavy")) or egraph_metrics.equivalence_density >= 2:
        candidate_costs["egraph_hybrid"] *= 0.78
        reasons.append("equivalence_bias")

    signals = {
        "typed_hypergraph": hyper_signal,
        "egraph_hybrid": egraph_signal,
        "structure_graph": graph_signal,
    }
    best_primitive = min(PRIMITIVES, key=lambda primitive: candidate_costs[primitive])
    best_signal = signals[best_primitive]
    reason = "lowest_estimated_cost"
    if calibration and family:
        reason = "lowest_calibrated_cost"
    if reasons:
        reason = f"{reason}+{'+'.join(reasons)}"
    return RoutingDecision(best_primitive, reason, best_signal, candidate_costs)
