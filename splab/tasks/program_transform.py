"""Program-transform flavored rewrite cases."""

from __future__ import annotations

from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var
from splab.tasks.task_case import TaskCase


def program_transform_cases() -> tuple[TaskCase, ...]:
    load = Var("load")
    filter_op = Var("filter")
    map_op = Var("map")
    join = Var("join")
    return (
        TaskCase(
            "pipeline_identity_cleanup",
            "program_transform",
            Mul(Add(load, Const(0)), Const(1)),
            load,
            {"flow_heavy": True, "graph_preferred": True, "pipeline_width": 4, "branch_factor": 2},
        ),
        TaskCase(
            "map_filter_fuse",
            "program_transform",
            Add(Add(map_op, filter_op), Const(0)),
            Add(filter_op, map_op),
            {"flow_heavy": True, "graph_preferred": True, "pipeline_width": 6, "branch_factor": 3},
        ),
        TaskCase(
            "join_const_fold",
            "program_transform",
            Mul(Add(join, Const(0)), Add(Const(2), Const(3))),
            Mul(Const(5), join),
            {"flow_heavy": True, "graph_preferred": True, "pipeline_width": 5, "branch_factor": 4},
        ),
        TaskCase(
            "wide_pipeline_balance",
            "program_transform",
            Add(Mul(Add(load, Const(0)), Const(1)), Add(map_op, filter_op)),
            Add(Add(filter_op, load), map_op),
            {"flow_heavy": True, "graph_preferred": True, "pipeline_width": 8, "branch_factor": 5},
        ),
    )
