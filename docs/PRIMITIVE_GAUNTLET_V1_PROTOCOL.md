# Primitive Gauntlet V1 Protocol

## Purpose

Primitive Gauntlet V1 is a pre-registered qualification harness for testing whether
structure-native execution primitives can deliver a reproducible systems-level advantage
over dense matrix encodings on semantically matched workloads.

This protocol is intentionally narrower than an architecture claim. Passing it does not
establish superiority over Transformers, GNNs, e-graphs, theorem provers, or general
matrix computation.

## Research question

For three matched workload families, does a structure-native representation preserve the
same output while reducing measured latency and peak accelerator memory relative to a
dense matrix representation as problem size grows?

The workload families are:

1. **graph propagation** — sparse edge-index aggregation versus a dense adjacency matrix;
2. **hypergraph reduction** — indexed hyperedge reduction versus a dense incidence matrix;
3. **equivalence-class reduction** — class-index aggregation/broadcast versus a dense
   membership matrix.

Each native/dense pair computes the same deterministic function from the same logical
problem specification.

## Frozen profiles

The runner exposes four profiles:

| Profile | Sizes |
| --- | --- |
| `smoke` | 64, 128 |
| `l4` | 512, 2048, 4096 |
| `a100` | 1024, 4096, 8192 |
| `g4` | 1024, 4096, 8192, 16384 |

Default feature width is 64. Graph fanout, hyperedge arity, and equivalence-class size are
all fixed at 8 unless explicitly overridden in a new protocol version.

## Measurement procedure

For every workload/size pair:

1. Build and benchmark the dense backend in isolation.
2. Build and benchmark the native backend in isolation.
3. Repeat in reverse order.
4. Aggregate timing samples from both measurement blocks.
5. Record median and p95 latency.
6. On CUDA, record peak allocated memory separately for each backend.
7. Compare the final outputs and record maximum absolute error.

Backend setup/allocation is excluded from latency but included in the peak-memory
measurement. Warm-up runs are excluded from timing.

## Correctness gate

Every tested row must satisfy:

```text
max_abs_error <= 1e-4
```

A performance result with a failed correctness gate is invalid.

## Phase-B promotion gate

This gate decides whether a primitive family is worth a custom Triton/CUDA implementation.
It is not a publication claim.

A family qualifies when **both of its two largest tested sizes** satisfy:

```text
native median latency speedup >= 1.20x
native peak memory <= 50% of dense peak memory
```

The overall V1 program is `PROMOTE` only when:

- all correctness gates pass;
- at least two of the three workload families qualify; and
- no family is catastrophically slower at its largest size
  (`native speedup < 0.50x`).

If CUDA memory telemetry is unavailable, the runner reports the memory gate as
`not_measured`; CPU smoke tests can validate semantics but cannot promote the program.

## Claim boundary

A V1 pass supports only this statement:

> Under the recorded hardware/software environment and the frozen V1 workload
> definitions, the tested structure-native implementations satisfied the registered
> correctness and qualification gates relative to their dense encodings.

A pass does **not** establish:

- end-to-end neural-network superiority;
- Transformer replacement;
- asymptotic optimality;
- general graph/hypergraph/e-graph superiority;
- a custom-kernel advantage before the Triton/CUDA phase exists.

## Required artifact fields

Every result artifact must record:

- git commit/branch when available;
- timestamp;
- Python version;
- PyTorch version;
- CUDA version;
- device name, capability, and total memory;
- profile and runner parameters;
- per-row correctness, latency, speedup, and memory;
- the final promotion decision.

## Next phase

If V1 promotes, Phase B should implement custom Triton/CUDA kernels without changing the
semantic workload definitions or the V1 correctness oracle. If V1 does not promote, inspect
the failure mode before changing primitives or workloads; a revised protocol must use a new
version identifier.
