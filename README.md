# structure-primitive-lab

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

## Current MVP

The current MVP includes:

- symbolic arithmetic rewrite tasks
- structure extraction into three candidate representations
- structural complexity scoring
- primitive routing
- correctness and synthetic cost comparison against baselines

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
- while selecting cheaper structure-aware execution paths on average

## Next steps

1. Add program-transform tasks.
2. Add proof-state transition tasks.
3. Add real GPU lowering experiments through existing tensor backends.
4. Replace synthetic cost models with measured kernels.
