"""CLI runner for Primitive Kernel Gauntlet V2."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from splab.kernel_gauntlet.bench import benchmark_row
from splab.kernel_gauntlet.spec import FAMILIES, g4_specs

PROTOCOL = "primitive-kernel-gauntlet-v2"


def _environment(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    try:
        import triton
        triton_version = getattr(triton, "__version__", "unknown")
    except Exception:
        triton_version = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gib": round(props.total_memory / (1024.0 ** 3), 3),
    }


def promotion_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    all_correct = all(bool(row["correct"]) for row in rows)
    family_results: dict[str, dict[str, object]] = {}
    qualified_count = 0
    catastrophic = False

    for family in FAMILIES:
        group = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: int(row["size"]),
        )
        largest_two = group[-2:]
        speed_gate = len(largest_two) == 2 and all(
            float(row["triton_speedup_vs_torch_native"]) >= 1.10
            for row in largest_two
        )
        memory_gate = len(largest_two) == 2 and all(
            float(row["triton_to_torch_peak_memory_ratio"]) <= 1.10
            for row in largest_two
        )
        qualifies = speed_gate and memory_gate
        qualified_count += int(qualifies)
        if group:
            catastrophic = catastrophic or float(group[-1]["triton_speedup_vs_torch_native"]) < 0.80
        family_results[family] = {
            "largest_two_sizes": [int(row["size"]) for row in largest_two],
            "speed_gate": speed_gate,
            "memory_gate": memory_gate,
            "qualifies": qualifies,
        }

    promote = all_correct and qualified_count >= 2 and not catastrophic
    return {
        "all_correct": all_correct,
        "qualified_family_count": qualified_count,
        "catastrophic_slowdown": catastrophic,
        "family_results": family_results,
        "decision": "PROMOTE" if promote else "HOLD",
    }


def run(*, warmup: int = 10, repeats: int = 30) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("V2 requires a CUDA runtime")
    device = torch.device("cuda")
    rows = [
        benchmark_row(spec, device, warmup=warmup, repeats=repeats)
        for spec in g4_specs()
    ]
    return {
        "protocol": PROTOCOL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "g4",
        "warmup": warmup,
        "repeats_per_abba_block": repeats,
        "environment": _environment(device),
        "summary": promotion_summary(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run(warmup=max(1, args.warmup), repeats=max(1, args.repeats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
