from splab.kernel_gauntlet.runner import promotion_summary
from splab.kernel_gauntlet.spec import FAMILIES, G4_SIZES, g4_specs


def test_v2_g4_profile_is_frozen() -> None:
    specs = g4_specs()
    assert {spec.family for spec in specs} == set(FAMILIES)
    assert {spec.size for spec in specs} == set(G4_SIZES)
    assert len(specs) == len(FAMILIES) * len(G4_SIZES)


def test_promotion_gate_requires_two_qualified_families() -> None:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for size in G4_SIZES:
            rows.append({
                "family": family,
                "size": size,
                "correct": True,
                "triton_speedup_vs_torch_native": 1.20 if family_index < 2 else 1.00,
                "triton_to_torch_peak_memory_ratio": 1.00,
            })
    summary = promotion_summary(rows)
    assert summary["decision"] == "PROMOTE"
    assert summary["qualified_family_count"] == 2
