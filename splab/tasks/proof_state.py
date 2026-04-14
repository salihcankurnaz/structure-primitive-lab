"""Proof-state flavored rewrite cases."""

from __future__ import annotations

from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var
from splab.tasks.task_case import TaskCase


def proof_state_cases() -> tuple[TaskCase, ...]:
    goal = Var("goal")
    premise_a = Var("premise_a")
    premise_b = Var("premise_b")
    tactic = Var("tactic")
    return (
        TaskCase(
            "premise_cleanup",
            "proof_state",
            Add(Mul(premise_a, Const(1)), Const(0)),
            premise_a,
            {"relation_arity": 3, "hypergraph_preferred": True, "context_width": 4, "dependency_fanout": 2},
        ),
        TaskCase(
            "goal_context_fold",
            "proof_state",
            Mul(Add(goal, Const(0)), Add(Const(2), Const(3))),
            Mul(Const(5), goal),
            {"relation_arity": 5, "hypergraph_preferred": True, "context_width": 8, "dependency_fanout": 4},
        ),
        TaskCase(
            "tactic_premise_mix",
            "proof_state",
            Add(Mul(tactic, Const(1)), Add(premise_b, Const(0))),
            Add(premise_b, tactic),
            {"relation_arity": 4, "hypergraph_preferred": True, "equivalence_heavy": True, "context_width": 6, "dependency_fanout": 3},
        ),
        TaskCase(
            "goal_premise_weave",
            "proof_state",
            Add(Mul(Add(goal, Const(0)), Const(1)), Add(premise_a, premise_b)),
            Add(Add(goal, premise_a), premise_b),
            {"relation_arity": 6, "hypergraph_preferred": True, "context_width": 10, "dependency_fanout": 5},
        ),
    )
