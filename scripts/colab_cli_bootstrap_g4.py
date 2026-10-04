"""Remote bootstrap for Primitive Gauntlet V1 via the official Colab CLI.

This file is intended to be transmitted with:
  colab run --gpu G4 --keep -s primitive-g4 --timeout 3600 scripts/colab_cli_bootstrap_g4.py

The remote VM clones the frozen research branch, installs the package, executes
the G4 gauntlet, and copies deterministic download targets under /content.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/salihcankurnaz/structure-primitive-lab.git"
BRANCH = "research/primitive-gauntlet-v1"
REMOTE_ROOT = Path("/content/structure-primitive-lab")
REMOTE_OUT = Path("/content/primitive_gauntlet_out")


def run_cmd(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(list(args), cwd=str(cwd) if cwd else None)


def main() -> None:
    if REMOTE_ROOT.exists():
        shutil.rmtree(REMOTE_ROOT)
    if REMOTE_OUT.exists():
        shutil.rmtree(REMOTE_OUT)
    REMOTE_OUT.mkdir(parents=True, exist_ok=True)

    run_cmd("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REMOTE_ROOT))
    run_cmd(sys.executable, "-m", "pip", "install", "-e", ".", cwd=REMOTE_ROOT)
    run_cmd(
        sys.executable,
        "scripts/run_primitive_gauntlet_v1.py",
        "--profile",
        "g4",
        "--device",
        "cuda",
        "--warmup",
        "5",
        "--repeats",
        "10",
        "--output-dir",
        str(REMOTE_OUT / "raw"),
        cwd=REMOTE_ROOT,
    )

    result_zips = sorted((REMOTE_OUT / "raw").glob("RESULT_PRIMITIVE_GAUNTLET_V1_*.zip"))
    source_zips = sorted((REMOTE_OUT / "raw").glob("SOURCE_PRIMITIVE_GAUNTLET_V1_*.zip"))
    reports = sorted((REMOTE_OUT / "raw").glob("PRIMITIVE_GAUNTLET_V1_G4_*.json"))
    metadata = sorted((REMOTE_OUT / "raw").glob("RUN_METADATA_*.json"))

    if not result_zips or not source_zips or not reports:
        raise RuntimeError("Expected gauntlet artifacts were not produced")

    shutil.copy2(result_zips[-1], REMOTE_OUT / "RESULT_PRIMITIVE_GAUNTLET_V1.zip")
    shutil.copy2(source_zips[-1], REMOTE_OUT / "SOURCE_PRIMITIVE_GAUNTLET_V1.zip")
    shutil.copy2(reports[-1], REMOTE_OUT / "PRIMITIVE_GAUNTLET_V1_G4.json")
    if metadata:
        shutil.copy2(metadata[-1], REMOTE_OUT / "RUN_METADATA.json")

    print("\nCOLAB_CLI_GAUNTLET_COMPLETE", flush=True)
    print(f"RESULT={REMOTE_OUT / 'RESULT_PRIMITIVE_GAUNTLET_V1.zip'}", flush=True)
    print(f"SOURCE={REMOTE_OUT / 'SOURCE_PRIMITIVE_GAUNTLET_V1.zip'}", flush=True)
    print(f"REPORT={REMOTE_OUT / 'PRIMITIVE_GAUNTLET_V1_G4.json'}", flush=True)


if __name__ == "__main__":
    main()
