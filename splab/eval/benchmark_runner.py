"""Benchmark runner for the first structure-heavy task."""

from __future__ import annotations

from splab.baselines.gnn_like import run_gnn_like_baseline
from splab.baselines.heuristic_solver import run_heuristic_baseline
from splab.baselines.transformer_like import run_transformer_like_baseline
from splab.execution.calibration import PRIMITIVES
from splab.execution.advisor import advise_primitive
from splab.execution.engine import run_engine
from splab.gpu.lowering import plan_gpu_lowering
from splab.gpu.runtime import measure_all_primitives, measure_runtime
from splab.tasks.constraint_native import constraint_native_cases
from splab.tasks.graph_native import graph_native_cases
from splab.tasks.hypergraph_native import hypergraph_native_cases
from splab.tasks.program_transform import program_transform_cases
from splab.tasks.proof_state import proof_state_cases
from splab.tasks.symbolic_rewrite import pretty_expr, symbolic_cases


def run() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_cases = (
        tuple(symbolic_cases())
        + tuple(program_transform_cases())
        + tuple(proof_state_cases())
        + tuple(hypergraph_native_cases())
        + tuple(constraint_native_cases())
        + tuple(graph_native_cases())
    )
    for case in all_cases:
        advice = advise_primitive(case)
        engine = run_engine(case)
        heuristic = run_heuristic_baseline(case.expr)
        gnn = run_gnn_like_baseline(case.expr)
        transformer = run_transformer_like_baseline(case.expr)
        lowering = plan_gpu_lowering(engine)
        runtime = measure_runtime(engine)
        primitive_ms = measure_all_primitives(engine)
        dense_ms = primitive_ms["dense_attention"]
        oracle_primitive = min(PRIMITIVES, key=lambda primitive: primitive_ms[primitive])
        route_regret_ms = runtime.mean_ms - primitive_ms[oracle_primitive]
        route_speedup_vs_dense = (dense_ms / runtime.mean_ms) if runtime.mean_ms > 0 else 0.0
        rows.append(
            {
                "case": case.name,
                "family": case.family,
                "route": engine.decision.primitive,
                "advisor_primitive": advice.primitive if advice else None,
                "advisor_confidence": advice.confidence if advice else None,
                "advisor_reason": advice.reason if advice else None,
                "reason": engine.decision.reason,
                "metadata": case.metadata,
                "correct": pretty_expr(engine.rewrite.expr) == pretty_expr(case.expected),
                "expected": pretty_expr(case.expected),
                "engine": pretty_expr(engine.rewrite.expr),
                "heuristic_cost": heuristic.estimated_cost,
                "gnn_cost": gnn.estimated_cost,
                "transformer_cost": transformer.estimated_cost,
                "engine_cost": engine.estimated_cost,
                "invariant": engine.invariant.score,
                "gpu_backend": lowering.backend,
                "measured_ms": runtime.mean_ms,
                "dense_baseline_ms": round(dense_ms, 4),
                "speedup_vs_dense": round(route_speedup_vs_dense, 4),
                "all_primitive_ms": {key: round(value, 4) for key, value in primitive_ms.items()},
                "oracle_primitive": oracle_primitive,
                "route_matches_oracle": engine.decision.primitive == oracle_primitive,
                "route_regret_ms": round(route_regret_ms, 4),
                "candidate_costs": engine.decision.candidate_costs,
                "device": runtime.device,
            }
        )
    return rows


if __name__ == "__main__":
    rows = run()
    print("structure-primitive-lab benchmark")
    for row in rows:
        print(
            f"  family={row['family']}, case={row['case']}, route={row['route']}, reason={row['reason']}, "
            f"advisor={row['advisor_primitive']}, "
            f"correct={row['correct']}, engine_cost={row['engine_cost']}, "
            f"heuristic_cost={row['heuristic_cost']}, gnn_cost={row['gnn_cost']}, "
            f"transformer_cost={row['transformer_cost']}, gpu_backend={row['gpu_backend']}, "
            f"measured_ms={row['measured_ms']}, dense_baseline_ms={row['dense_baseline_ms']}, "
            f"speedup_vs_dense={row['speedup_vs_dense']}, oracle={row['oracle_primitive']}, "
            f"route_matches_oracle={row['route_matches_oracle']}, route_regret_ms={row['route_regret_ms']}, "
            f"device={row['device']}"
        )
