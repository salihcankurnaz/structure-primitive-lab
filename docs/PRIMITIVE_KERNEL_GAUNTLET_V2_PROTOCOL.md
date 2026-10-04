# Primitive Kernel Gauntlet V2 Protocol

## Why V2 exists

Primitive Gauntlet V1 passed on a Colab G4-class runtime. That result establishes only
that the registered structure-native PyTorch implementations beat deliberately dense
matrix encodings at large sizes while preserving the same outputs.

That is not yet a novel primitive result: sparse/indexed computation is expected to beat
an O(N^2) dense encoding when the logical structure has fixed fanout/arity.

V2 therefore raises the bar. The primary comparison is now:

```text
custom Triton structure kernel
        vs
best V1 PyTorch structure-native implementation
```

Dense baselines are retained only as context at sizes where they are inexpensive enough.

## Frozen workloads

The semantic definitions are unchanged from V1:

1. graph propagation: each destination sums the previous `fanout` cyclic nodes;
2. hypergraph reduction: each hyperedge averages `arity` cyclic member nodes;
3. equivalence reduction: every node receives the mean feature vector of its fixed-size
   equivalence class.

Default feature width is 64 and fanout/arity/class-size are 8.

## G4 profile

V2 G4 sizes are:

```text
8192, 16384, 32768, 65536, 131072
```

The dense context baseline is measured only through size 16384. The promotion decision
does not depend on dense performance.

## Measurement

For each family/size:

- Triton is compiled before the measured block.
- PyTorch-native and Triton are measured in ABBA order.
- Warm-up samples are excluded.
- Median and p95 latency are recorded.
- Peak CUDA allocated memory is recorded separately.
- Triton output is compared against PyTorch-native output.

## Correctness gate

Every row must satisfy:

```text
max_abs_error <= 1e-4
```

A performance number from an incorrect kernel is invalid.

## Phase-C promotion gate

For a family to qualify, both of its two largest sizes (65536 and 131072 on G4) must
satisfy:

```text
Triton / PyTorch-native median-latency speedup >= 1.10x
Triton peak memory / PyTorch-native peak memory <= 1.10x
```

The V2 program is `PROMOTE` only if:

- all correctness gates pass;
- at least two of the three families qualify;
- no family is catastrophically slower at the largest size
  (`Triton speedup < 0.80x`).

Otherwise the decision is `HOLD`.

## Claim boundary

A V2 pass supports only:

> On the recorded hardware/software environment and frozen V2 workloads, the custom
> Triton structure kernels met the registered correctness and performance gates relative
> to the V1 PyTorch structure-native implementations.

It does not establish:

- novelty of sparse computation;
- superiority over optimized PyG/DGL/torch-scatter libraries;
- Transformer or GNN replacement;
- end-to-end model quality gains;
- general asymptotic optimality.

## What a pass unlocks

A V2 pass unlocks Phase C: integration into real structure-heavy workloads, comparison
against optimized external sparse/graph libraries, and end-to-end quality/performance
experiments. Only Phase C can start testing whether the primitive is useful beyond this
controlled microbenchmark.
