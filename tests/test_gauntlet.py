import torch

from splab.gauntlet.bench import benchmark_pair
from splab.gauntlet.runner import _promotion_summary
from splab.gauntlet.spec import FAMILIES, WorkloadSpec, profile_specs


def test_smoke_profile_covers_all_families() -> None:
    specs = profile_specs("smoke", feature_dim=8, fanout=2, arity=3, class_size=4)
    assert {spec.family for spec in specs} == set(FAMILIES)
    assert {spec.size for spec in specs} == {64, 128}


def test_matched_workloads_are_correct_on_cpu() -> None:
    device = torch.device("cpu")
    for family in FAMILIES:
        spec = WorkloadSpec(
            family=family,
            size=32,
            feature_dim=8,
            fanout=2,
            arity=3,
            class_size=4,
        )
        row = benchmark_pair(spec, device, warmup=0, repeats=1)
        assert row["correct"] is True
        assert float(row["max_abs_error"]) <= 1e-4


def test_cpu_results_cannot_promote_without_memory_telemetry() -> None:
    rows = []
    for family in FAMILIES:
        for size in (32, 64):
            rows.append(
                {
                    "family": family,
                    "size": size,
                    "correct": True,
                    "speedup_vs_dense": 2.0,
                    "native_to_dense_peak_memory_ratio": None,
                }
            )
    summary = _promotion_summary(rows)
    assert summary["decision"] == "HOLD"
    assert summary["qualified_family_count"] == 0
