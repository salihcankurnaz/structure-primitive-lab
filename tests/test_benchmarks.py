from splab.eval.benchmark_runner import run
from splab.eval.reports import summarize


def test_benchmark_runner_is_correct_on_all_cases() -> None:
    rows = run()
    assert rows
    assert all(bool(row["correct"]) for row in rows)
    assert all(float(row["measured_ms"]) >= 0.0 for row in rows)
    assert all(float(row["dense_baseline_ms"]) >= 0.0 for row in rows)
    assert all(isinstance(row["all_primitive_ms"], dict) for row in rows)
    assert all("oracle_primitive" in row for row in rows)
    assert all("route_regret_ms" in row for row in rows)
    assert all("advisor_primitive" in row for row in rows)


def test_summary_reports_cost_gap() -> None:
    summary = summarize()
    assert summary["case_count"] == 29
    assert summary["accuracy"] == 1.0
    assert summary["avg_engine_cost"] < summary["avg_transformer_cost"]
    assert summary["avg_measured_ms"] >= 0.0
    assert summary["avg_dense_baseline_ms"] >= 0.0
    assert 0.0 <= summary["oracle_route_accuracy"] <= 1.0
    assert abs(summary["avg_route_regret_ms"]) < 1.0
    assert "program_transform" in summary["family_routes"]
    assert "proof_state" in summary["family_routes"]
    assert "hypergraph_native" in summary["family_routes"]
    assert "constraint_native" in summary["family_routes"]
    assert "graph_native" in summary["family_routes"]
    assert "family_wins" in summary
    assert "family_frontier" in summary
