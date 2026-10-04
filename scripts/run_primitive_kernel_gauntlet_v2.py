"""Run Primitive Kernel Gauntlet V2 and create source/result bundles."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from splab.kernel_gauntlet.runner import run


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def zip_paths(out: Path, root: Path, paths: list[Path]) -> None:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file() and "__pycache__" not in item.parts:
                        z.write(item, item.relative_to(root))
            elif path.exists():
                z.write(path, path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/primitive_kernel_gauntlet_v2"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = run(warmup=args.warmup, repeats=args.repeats)
    payload["git_commit"] = git_commit(root)

    report = args.output_dir / f"PRIMITIVE_KERNEL_GAUNTLET_V2_G4_{stamp}.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metadata = args.output_dir / f"RUN_METADATA_V2_{stamp}.json"
    metadata.write_text(json.dumps({
        "protocol": payload["protocol"],
        "git_commit": payload["git_commit"],
        "timestamp_utc": payload["timestamp_utc"],
        "decision": payload["summary"]["decision"],
        "all_correct": payload["summary"]["all_correct"],
        "qualified_family_count": payload["summary"]["qualified_family_count"],
    }, indent=2), encoding="utf-8")

    source_zip = args.output_dir / f"SOURCE_PRIMITIVE_KERNEL_GAUNTLET_V2_{stamp}.zip"
    zip_paths(source_zip, root, [
        root / "splab" / "kernels",
        root / "splab" / "kernel_gauntlet",
        root / "scripts" / "run_primitive_kernel_gauntlet_v2.py",
        root / "docs" / "PRIMITIVE_KERNEL_GAUNTLET_V2_PROTOCOL.md",
        root / "pyproject.toml",
    ])

    result_zip = args.output_dir / f"RESULT_PRIMITIVE_KERNEL_GAUNTLET_V2_{stamp}.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(report, report.name)
        z.write(metadata, metadata.name)
        z.write(root / "docs" / "PRIMITIVE_KERNEL_GAUNTLET_V2_PROTOCOL.md", "PRIMITIVE_KERNEL_GAUNTLET_V2_PROTOCOL.md")

    print(f"decision={payload['summary']['decision']}")
    print(f"report={report}")
    print(f"source_zip={source_zip}")
    print(f"result_zip={result_zip}")


if __name__ == "__main__":
    main()
