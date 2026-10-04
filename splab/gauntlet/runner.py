"""CLI runner for Primitive Gauntlet V1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from splab.gauntlet.bench import benchmark_pair
from splab.gauntlet.spec import FAMILIES, available_profiles, profile_specs

PROTOCOL_VERSION = "primitive-gauntlet-v1"


def _device_info(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gib": round(props.total_memory / (1024.0 ** 3), 3),
            }
        )
    return payload


def _promotion_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    all_correct = all(bool(row["correct"]) for row in rows)
    family_results: dict[str, dict[str, object]] = {}
    qualified = 0
    catastrophic = False

    for family in FAMILIES:
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: int(row["size"]),
        )
        largest_two = family_rows[-2:] if len(family_rows) >= 2 else family_rows
        speed_gate = bool(largest_two) and all(
            isinstance(row["speedup_vs_dense"], (int, float))
            and float(row["speedup_vs_dense"]) >= 1.20
            for row in largest_two
        )
        memory_values = [row["native_to_dense_peak_memory_ratio"] for row in largest_two]
        memory_measured = bool(memory_values) and all(isinstance(value, (int, float)) for value in memory_values)
        memory_gate = memory_measured and all(float(value) <= 0.50 for value in memory_values)

        largest = family_rows[-1] if family_rows else None
        if largest is not None and isinstance(largest["speedup_vs_dense"], (int, float)):
            catastrophic = catastrophic or float(largest["speedup_vs_dense"]) < 0.50

        family_qualifies = speed_gate and memory_gate
        qualified += int(family_qualifies)
        family_results[family] = {
            "speed_gate": speed_gate,
            "memory_gate": memory_gate if memory_measured else "not_measured",
            "qualifies": family_qualifies,
            "largest_two_sizes": [int(row["size"]) for row in largest_two],
        }

    promote = all_correct and qualified >= 2 and not catastrophic
    return {
        "all_correct": all_correct,
        "qualified_family_count": qualified,
        "catastrophic_slowdown": catastrophic,
        "family_results": family_results,
        "decision": "PROMOTE" if promote else "HOLD",
    }


def run(
    profile: str,
    *,
    device: str | None = None,
    warmup: int = 5,
    repeats: int = 10,
) -> dict[str, object]:
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    rows: list[dict[str, object]] = []
    for spec in profile_specs(profile):
        rows.append(
            benchmark_pair(
                spec,
                selected,
                warmup=warmup,
                repeats=repeats,
            )
        )

    return {
        "protocol": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "warmup": warmup,
        "repeats_per_abba_block": repeats,
        "environment": _device_info(selected),
        "summary": _promotion_summary(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Primitive Gauntlet V1")
    parser.add_argument("--profile", choices=available_profiles(), default="smoke")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run(
        args.profile,
        device=args.device,
        warmup=max(0, args.warmup),
        repeats=max(1, args.repeats),
    )

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = Path("experiments") / "reports" / f"primitive_gauntlet_v1_{args.profile}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    print(f"wrote: {output}")
    print(f"decision: {summary['decision']}")
    print(f"all_correct: {summary['all_correct']}")
    print(f"qualified_family_count: {summary['qualified_family_count']}")


if __name__ == "__main__":
    main()
