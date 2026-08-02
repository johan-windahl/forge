"""The kernel: durable state, the task graph, scheduling and recovery.

Nothing in the kernel knows what a language model is. It knows about *nodes*
that need doing, *events* that record what happened, *leases* that stop two
workers doing the same thing, and *checkpoints* that let a run be rewound. That
separation is what makes the platform survivable: the parts that fail most often
(models, networks, browsers) sit above a layer that has no opinion about them.
"""

from .events import Event, EventType
from .graph import Node, NodeKind, NodeStatus, TaskGraph
from .ledger import Ledger

__all__ = [
    "Event",
    "EventType",
    "Ledger",
    "Node",
    "NodeKind",
    "NodeStatus",
    "TaskGraph",
]
