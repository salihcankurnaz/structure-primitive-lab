"""Simple report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from splab.eval.benchmark_runner import run


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    family_routes: dict[str, list[str]] = {}
    for row in rows:
        family_routes.setdefault(str(row["family"]), []).append(str(row["route"]))
    family_speedups: dict[str, list[float]] = {}
    for row in rows:
        family_speedups.setdefault(str(row["family"]), []).append(float(row["speedup_vs_dense"]))
    family_oracles: dict[str, list[str]] = {}
    for row in rows:
        family_oracles.setdefault(str(row["family"]), []).append(str(row["oracle_primitive"]))
    family_wins: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row["family"])
        oracle = str(row["oracle_primitive"])
        family_wins.setdefault(family, {}).setdefault(oracle, 0)
        family_wins[family][oracle] += 1
    family_frontier: dict[str, dict[str, float]] = {}
    for row in rows:
        family = str(row["family"])
        primitive_times = row.get("all_primitive_ms", {})
        if not isinstance(primitive_times, dict):
            continue
        family_frontier.setdefault(family, {})
        for primitive, value in primitive_times.items():
            if primitive == "dense_attention":
                continue
            family_frontier[family].setdefault(str(primitive), 0.0)
            family_frontier[family][str(primitive)] += float(value)
    for family, primitive_totals in family_frontier.items():
        count = max(1, sum(1 for row in rows if str(row["family"]) == family))
        family_frontier[family] = {
            primitive: round(total / count, 4) for primitive, total in primitive_totals.items()
        }
    return {
        "case_count": len(rows),
        "accuracy": mean(1.0 if row["correct"] else 0.0 for row in rows),
        "avg_engine_cost": mean(float(row["engine_cost"]) for row in rows),
        "avg_transformer_cost": mean(float(row["transformer_cost"]) for row in rows),
        "avg_measured_ms": mean(float(row["measured_ms"]) for row in rows),
        "avg_dense_baseline_ms": mean(float(row["dense_baseline_ms"]) for row in rows),
        "oracle_route_accuracy": mean(1.0 if row["route_matches_oracle"] else 0.0 for row in rows),
        "avg_route_regret_ms": mean(float(row["route_regret_ms"]) for row in rows),
        "routes": sorted({str(row["route"]) for row in rows}),
        "family_routes": {family: sorted(set(routes)) for family, routes in family_routes.items()},
        "family_oracles": {family: sorted(set(routes)) for family, routes in family_oracles.items()},
        "family_wins": family_wins,
        "family_frontier": family_frontier,
        "family_speedups": {family: round(mean(speedups), 4) for family, speedups in family_speedups.items()},
    }


def summarize() -> dict[str, object]:
    return _summarize_rows(run())


def summarize_report(base_dir: Path | None = None) -> dict[str, object] | None:
    root = base_dir or Path(__file__).resolve().parents[2]
    report_path = root / "experiments" / "reports" / "structure_primitive_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    rows = payload.get("rows")
    if isinstance(rows, list):
        return _summarize_rows(rows)
    return None


def write_report(base_dir: Path | None = None, bootstrap_rounds: int = 2) -> Path:
    root = base_dir or Path(__file__).resolve().parents[2]
    out_dir = root / "experiments" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structure_primitive_report.json"
    rounds = max(1, bootstrap_rounds)
    last_rows: list[dict[str, object]] = []
    last_summary: dict[str, object] = {}
    for _ in range(rounds):
        last_rows = run()
        family_routes: dict[str, list[str]] = {}
        family_speedups: dict[str, list[float]] = {}
        family_oracles: dict[str, list[str]] = {}
        family_wins: dict[str, dict[str, int]] = {}
        for row in last_rows:
            family = str(row["family"])
            route = str(row["route"])
            oracle = str(row["oracle_primitive"])
            family_routes.setdefault(family, []).append(route)
            family_speedups.setdefault(family, []).append(float(row["speedup_vs_dense"]))
            family_oracles.setdefault(family, []).append(oracle)
            family_wins.setdefault(family, {}).setdefault(oracle, 0)
            family_wins[family][oracle] += 1
        last_summary = _summarize_rows(last_rows)
        payload = {
            "summary": last_summary,
            "rows": last_rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
