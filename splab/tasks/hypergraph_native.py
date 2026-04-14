"""Hypergraph-native structure-heavy task family."""

from __future__ import annotations

from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var
from splab.tasks.task_case import TaskCase


def hypergraph_native_cases() -> tuple[TaskCase, ...]:
    bundle = Var("bundle")
    clause = Var("clause")
    witness = Var("witness")
    context = Var("context")
    return (
        TaskCase(
            "bundle_cleanup",
            "hypergraph_native",
            Add(Mul(bundle, Const(1)), Const(0)),
            bundle,
            {
                "hypergraph_preferred": True,
                "hyperedge_native": True,
                "relation_arity": 8,
                "context_width": 14,
                "dependency_fanout": 2,
            },
        ),
        TaskCase(
            "clause_context_fold",
            "hypergraph_native",
            Mul(Add(clause, Const(0)), Add(Const(2), Const(3))),
            Mul(Const(5), clause),
            {
                "hypergraph_preferred": True,
                "hyperedge_native": True,
                "relation_arity": 10,
                "context_width": 18,
                "dependency_fanout": 2,
            },
        ),
        TaskCase(
            "witness_bundle_mix",
            "hypergraph_native",
            Add(Mul(witness, Const(1)), Add(bundle, Const(0))),
            Add(bundle, witness),
            {
                "hypergraph_preferred": True,
                "hyperedge_native": True,
                "relation_arity": 9,
                "context_width": 16,
                "dependency_fanout": 3,
            },
        ),
        TaskCase(
            "context_clause_weave",
            "hypergraph_native",
            Add(Mul(Add(context, Const(0)), Const(1)), Add(clause, witness)),
            Add(Add(clause, context), witness),
            {
                "hypergraph_preferred": True,
                "hyperedge_native": True,
                "relation_arity": 12,
                "context_width": 22,
                "dependency_fanout": 3,
            },
        ),
    )
