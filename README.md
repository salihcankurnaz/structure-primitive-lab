# structure-primitive-lab

> **Research status:** early-stage prototype. Current comparisons use correctness checks and synthetic cost models; the repository does **not** yet establish a measured GPU speedup or superiority over Transformer/GNN implementations.

Post-matrix, post-attention primitives for structure-heavy computation.

This repository is a research harness for testing whether structure-aware
representations can beat dense, matrix-first computation on reasoning-heavy
tasks.

The first version focuses on a small but real end-to-end pipeline:

- three candidate structure primitives
  - typed hypergraph
  - e-graph hybrid
  - structure graph
- one shared symbolic rewrite task
- one routed execution engine
- three comparison baselines
  - transformer-like dense scoring
  - GNN-like local propagation
  - heuristic rewrite baseline

## Why this repo exists

Some workloads are currently expressed as matrices or dense attention mostly
because hardware and libraries are built that way. This repo tests a different
hypothesis:

> structure-heavy tasks may have a better native primitive than dense matrix
> computation.

This is a hypothesis under test, not a demonstrated general replacement for matrix or attention-based computation.

## Current MVP

The current MVP includes:

- symbolic arithmetic rewrite tasks
- structure extraction into three candidate representations
- structural complexity scoring
- primitive routing
- correctness and synthetic cost comparison against baselines

Synthetic cost comparisons are useful for exercising the routing and evaluation framework, but they should not be reported as hardware performance results. A performance claim requires measured kernels under a recorded hardware/software environment.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
python -m splab.eval.benchmark_runner
```

## Repo layout

```text
structure-primitive-lab/
  splab/
    representations/
    metrics/
    execution/
    gpu/
    tasks/
    baselines/
    eval/
  tests/
  experiments/
```

## Initial success criterion

The first success target is narrow and measurable:

- on symbolic rewrite workloads
- achieve equal correctness to dense and graph baselines
- while selecting lower-cost structure-aware execution paths under the current model

A later systems-level success criterion should replace estimated costs with wall-clock, memory, and throughput measurements from equivalent implementations.

## Reproducibility boundary

When reporting future benchmark results, record at minimum:

- repository commit;
- task/configuration;
- correctness oracle result;
- baseline implementation;
- hardware and software environment;
- warmup and timing procedure;
- raw measurements across repeated runs.

## Next steps

1. Add program-transform tasks.
2. Add proof-state transition tasks.
3. Add real GPU lowering experiments through existing tensor backends.
4. Replace synthetic cost models with measured kernels.


## Primitive Gauntlet V1

The first hardware qualification stage is frozen in
[`docs/PRIMITIVE_GAUNTLET_V1_PROTOCOL.md`](docs/PRIMITIVE_GAUNTLET_V1_PROTOCOL.md).
It compares semantically matched dense and structure-native implementations for graph
propagation, hypergraph reduction, and equivalence-class reduction.

CPU semantic smoke test:

```bash
python -m splab.gauntlet.runner --profile smoke --device cpu --warmup 0 --repeats 1
```

G4 qualification run:

```bash
python -m splab.gauntlet.runner --profile g4 --device cuda --warmup 5 --repeats 10
```

The runner writes a machine-readable JSON artifact containing environment metadata,
correctness, median/p95 latency, CUDA peak memory, per-size speedups, and the frozen
`PROMOTE`/`HOLD` gate. A pass is only a systems-qualification result for the registered
workloads; it is not evidence of general Transformer or GNN superiority.


For the Colab/G4 path with automatic source and result bundles:

```bash
python scripts/run_primitive_gauntlet_v1.py --profile g4 --device cuda --warmup 5 --repeats 10
```

This creates both `SOURCE_PRIMITIVE_GAUNTLET_V1_*.zip` and
`RESULT_PRIMITIVE_GAUNTLET_V1_*.zip` under `artifacts/primitive_gauntlet_v1/`.


### Official Colab CLI path

The official Google Colab CLI supports G4 allocation and one-shot remote execution.
On Linux/macOS (or WSL on Windows), from this repository:

```bash
python -m pip install google-colab-cli
bash scripts/run_colab_cli_g4.sh
```

The first `--auth=oauth2` invocation may ask for a browser authorization code. The launcher
keeps the named runtime only long enough to download the fixed result/source/report files,
then calls `colab stop` to release the VM.

The deterministic local outputs are:

```text
primitive_gauntlet_downloads/
  RESULT_PRIMITIVE_GAUNTLET_V1.zip
  SOURCE_PRIMITIVE_GAUNTLET_V1.zip
  PRIMITIVE_GAUNTLET_V1_G4.json
```
