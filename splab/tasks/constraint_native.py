"""Constraint-propagation flavored hypergraph-native tasks."""

from __future__ import annotations

from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var
from splab.tasks.task_case import TaskCase


def constraint_native_cases() -> tuple[TaskCase, ...]:
    domain = Var("domain")
    clause = Var("clause")
    support = Var("support")
    signal = Var("signal")
    return (
        TaskCase(
            "domain_cleanup",
            "constraint_native",
            Add(Mul(domain, Const(1)), Const(0)),
            domain,
            {
                "hypergraph_preferred": True,
                "constraint_native": True,
                "relation_arity": 10,
                "context_width": 18,
                "dependency_fanout": 4,
                "constraint_rounds": 6,
                "propagation_depth": 3,
            },
        ),
        TaskCase(
            "clause_support_fold",
            "constraint_native",
            Mul(Add(clause, Const(0)), Add(Const(2), Const(3))),
            Mul(Const(5), clause),
            {
                "hypergraph_preferred": True,
                "constraint_native": True,
                "relation_arity": 12,
                "context_width": 20,
                "dependency_fanout": 5,
                "constraint_rounds": 8,
                "propagation_depth": 4,
            },
        ),
        TaskCase(
            "signal_domain_mix",
            "constraint_native",
            Add(Mul(signal, Const(1)), Add(domain, Const(0))),
            Add(domain, signal),
            {
                "hypergraph_preferred": True,
                "constraint_native": True,
                "relation_arity": 9,
                "context_width": 16,
                "dependency_fanout": 4,
                "constraint_rounds": 7,
                "propagation_depth": 4,
            },
        ),
        TaskCase(
            "support_clause_weave",
            "constraint_native",
            Add(Mul(Add(support, Const(0)), Const(1)), Add(clause, signal)),
            Add(Add(clause, signal), support),
            {
                "hypergraph_preferred": True,
                "constraint_native": True,
                "relation_arity": 14,
                "context_width": 24,
                "dependency_fanout": 6,
                "constraint_rounds": 10,
                "propagation_depth": 5,
            },
        ),
    )
