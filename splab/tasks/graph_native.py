"""Graph-native dependency-flow tasks."""

from __future__ import annotations

from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var
from splab.tasks.task_case import TaskCase


def graph_native_cases() -> tuple[TaskCase, ...]:
    stream = Var("stream")
    filter_op = Var("filter")
    merge = Var("merge")
    sink = Var("sink")
    return (
        TaskCase(
            "stream_cleanup",
            "graph_native",
            Add(Mul(stream, Const(1)), Const(0)),
            stream,
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 10,
                "branch_factor": 6,
                "path_length": 5,
                "dependency_fanout": 2,
                "aggregation_bias": 1,
            },
        ),
        TaskCase(
            "filter_merge_fold",
            "graph_native",
            Mul(Add(filter_op, Const(0)), Add(Const(2), Const(3))),
            Mul(Const(5), filter_op),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 12,
                "branch_factor": 8,
                "path_length": 6,
                "dependency_fanout": 3,
                "aggregation_bias": 1,
            },
        ),
        TaskCase(
            "merge_stream_mix",
            "graph_native",
            Add(Mul(merge, Const(1)), Add(stream, Const(0))),
            Add(merge, stream),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 14,
                "branch_factor": 10,
                "path_length": 8,
                "dependency_fanout": 4,
                "aggregation_bias": 3,
            },
        ),
        TaskCase(
            "sink_dependency_weave",
            "graph_native",
            Add(Mul(Add(sink, Const(0)), Const(1)), Add(filter_op, merge)),
            Add(Add(filter_op, merge), sink),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 18,
                "branch_factor": 12,
                "path_length": 10,
                "dependency_fanout": 5,
                "aggregation_bias": 4,
            },
        ),
        TaskCase(
            "graph_chain_balance",
            "graph_native",
            Add(Mul(Add(stream, Const(0)), Const(1)), Add(filter_op, sink)),
            Add(Add(filter_op, sink), stream),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 20,
                "branch_factor": 14,
                "path_length": 12,
                "dependency_fanout": 6,
                "aggregation_bias": 5,
            },
        ),
        TaskCase(
            "graph_local_filter",
            "graph_native",
            Add(Mul(Add(filter_op, Const(0)), Const(1)), merge),
            Add(filter_op, merge),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 9,
                "branch_factor": 4,
                "path_length": 4,
                "dependency_fanout": 2,
                "aggregation_bias": 1,
            },
        ),
        TaskCase(
            "graph_aggregate_sink",
            "graph_native",
            Add(Mul(Add(sink, Const(0)), Const(1)), Add(stream, merge)),
            Add(Add(merge, sink), stream),
            {
                "graph_preferred": True,
                "graph_native": True,
                "flow_heavy": True,
                "pipeline_width": 22,
                "branch_factor": 16,
                "path_length": 14,
                "dependency_fanout": 7,
                "aggregation_bias": 6,
            },
        ),
    )
