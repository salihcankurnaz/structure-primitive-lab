#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-primitive-g4}"
LOCAL_OUT="${1:-./primitive_gauntlet_downloads}"

mkdir -p "$LOCAL_OUT"

echo "[1/5] Verifying official Colab CLI"
colab version

echo "[2/5] Starting G4 job on session: $SESSION"
colab --auth=oauth2 run \
  --gpu G4 \
  --keep \
  -s "$SESSION" \
  --timeout 3600 \
  scripts/colab_cli_bootstrap_g4.py

echo "[3/5] Runtime status"
colab --auth=oauth2 status -s "$SESSION"

echo "[4/5] Downloading frozen artifacts"
colab --auth=oauth2 download -s "$SESSION" \
  /content/primitive_gauntlet_out/RESULT_PRIMITIVE_GAUNTLET_V1.zip \
  "$LOCAL_OUT/RESULT_PRIMITIVE_GAUNTLET_V1.zip"
colab --auth=oauth2 download -s "$SESSION" \
  /content/primitive_gauntlet_out/SOURCE_PRIMITIVE_GAUNTLET_V1.zip \
  "$LOCAL_OUT/SOURCE_PRIMITIVE_GAUNTLET_V1.zip"
colab --auth=oauth2 download -s "$SESSION" \
  /content/primitive_gauntlet_out/PRIMITIVE_GAUNTLET_V1_G4.json \
  "$LOCAL_OUT/PRIMITIVE_GAUNTLET_V1_G4.json"

echo "[5/5] Releasing Colab runtime"
colab --auth=oauth2 stop -s "$SESSION"

echo "Done. Artifacts:"
ls -lh "$LOCAL_OUT"
