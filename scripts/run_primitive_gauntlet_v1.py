"""Colab-friendly Primitive Gauntlet V1 runner with reproducibility bundles."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from splab.gauntlet.runner import run


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_source_zip(root: Path, out_path: Path) -> None:
    include_roots = [
        root / "splab",
        root / "tests",
        root / "docs",
    ]
    include_files = [
        root / "README.md",
        root / "pyproject.toml",
    ]
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for base in include_roots:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(root))
        for path in include_files:
            if path.exists():
                archive.write(path, path.relative_to(root))


def _write_result_zip(report_path: Path, protocol_path: Path, metadata_path: Path, out_path: Path) -> None:
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        if protocol_path.exists():
            archive.write(protocol_path, protocol_path.name)
        archive.write(metadata_path, metadata_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and package Primitive Gauntlet V1")
    parser.add_argument("--profile", choices=("smoke", "l4", "a100", "g4"), default="g4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "primitive_gauntlet_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run(
        args.profile,
        device=args.device,
        warmup=max(0, args.warmup),
        repeats=max(1, args.repeats),
    )
    payload["git_commit"] = _git_commit(root)

    report_path = out_dir / f"PRIMITIVE_GAUNTLET_V1_{args.profile.upper()}_{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metadata = {
        "protocol": payload["protocol"],
        "profile": args.profile,
        "git_commit": payload["git_commit"],
        "timestamp_utc": payload["timestamp_utc"],
        "decision": payload["summary"]["decision"],
        "all_correct": payload["summary"]["all_correct"],
        "qualified_family_count": payload["summary"]["qualified_family_count"],
    }
    metadata_path = out_dir / f"RUN_METADATA_{stamp}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    source_zip = out_dir / f"SOURCE_PRIMITIVE_GAUNTLET_V1_{stamp}.zip"
    result_zip = out_dir / f"RESULT_PRIMITIVE_GAUNTLET_V1_{stamp}.zip"
    _write_source_zip(root, source_zip)
    _write_result_zip(
        report_path,
        root / "docs" / "PRIMITIVE_GAUNTLET_V1_PROTOCOL.md",
        metadata_path,
        result_zip,
    )

    print(f"decision={payload['summary']['decision']}")
    print(f"report={report_path}")
    print(f"source_zip={source_zip}")
    print(f"result_zip={result_zip}")


if __name__ == "__main__":
    main()
