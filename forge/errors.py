"""Error taxonomy.

The taxonomy exists so the scheduler can make *automatic* decisions without a
model in the loop: whether to retry, whether to back off, whether to escalate to
a stronger model, or whether to give up and surface the failure to a human.

Every error carries three orthogonal properties:

``retryable``
    Will the identical operation plausibly succeed if repeated? Network blips
    and rate limits are retryable; a syntax error in generated code is not.

``escalatable``
    Would a more capable model plausibly succeed? A failing test after three
    local-model attempts is escalatable; a missing binary on the host is not.

``transient``
    Is the fault in the environment rather than in the work? Transient faults do
    not count against a node's attempt budget, which prevents a flaky network
    from burning a task's retries.
"""

from __future__ import annotations

from typing import Any


class ForgeError(Exception):
    """Base class for every Forge-raised exception."""

    retryable: bool = False
    escalatable: bool = False
    transient: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
            "escalatable": self.escalatable,
            "transient": self.transient,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        detail = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({detail})"


# --------------------------------------------------------------------------
# Configuration and programming errors -- never retried.
# --------------------------------------------------------------------------


class ConfigError(ForgeError):
    """Malformed or missing configuration."""


class InvariantError(ForgeError):
    """A kernel invariant was violated. Indicates a bug in Forge itself."""


class NotSupported(ForgeError):
    """A capability was requested that this deployment does not provide."""


# --------------------------------------------------------------------------
# Storage / durability
# --------------------------------------------------------------------------


class LedgerError(ForgeError):
    """The append-only event ledger could not be read or written."""


class ConcurrencyError(LedgerError):
    """Optimistic concurrency check failed; caller should re-read and retry."""

    retryable = True
    transient = True


class LeaseLost(ForgeError):
    """A worker's lease on a node expired or was stolen while it was working."""

    retryable = True
    transient = True


# --------------------------------------------------------------------------
# Model layer
# --------------------------------------------------------------------------


class ModelError(ForgeError):
    """Base class for failures originating in a model provider."""


class ProviderUnavailable(ModelError):
    """Provider is unreachable: connection refused, DNS failure, 5xx."""

    retryable = True
    transient = True


class RateLimited(ModelError):
    """Provider returned 429 or an explicit throttling signal."""

    retryable = True
    transient = True

    def __init__(self, message: str, retry_after: float | None = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.retry_after = retry_after


class ContextOverflow(ModelError):
    """The assembled prompt exceeded the model's context window."""

    escalatable = True


class ReasoningBudgetExhausted(ModelError):
    """A reasoning model spent its whole output budget thinking.

    Specific to servers that separate chain-of-thought from the answer -- notably
    llama.cpp's ``reasoning_content``. The symptom is an empty answer with a full
    token count, which looks identical to a broken model unless you check. It is
    retryable with a larger output budget and is emphatically *not* a reason to
    escalate to a more expensive model.
    """

    retryable = True
    escalatable = False
    transient = False


class QuotaExhausted(ModelError):
    """A subscription-backed provider has used its allowance for the period.

    Distinct from :class:`BudgetExhausted`, which is about money. This is about
    plan rate limits, and the right response is to route elsewhere rather than
    to stop the run.
    """

    retryable = True
    transient = True


class MalformedOutput(ModelError):
    """The model produced output that failed schema validation after repair."""

    retryable = True
    escalatable = True


class BudgetExhausted(ModelError):
    """The configured token or cost ceiling has been reached."""


# --------------------------------------------------------------------------
# Execution / workspace
# --------------------------------------------------------------------------


class SandboxError(ForgeError):
    """The sandbox could not execute the requested command."""

    retryable = True
    transient = True


class CommandTimeout(SandboxError):
    """A command exceeded its wall-clock budget and was killed."""

    retryable = True
    transient = False


class GitError(ForgeError):
    """A git operation failed."""


class MergeConflict(GitError):
    """A node's isolated branch will not merge into the integrated tree.

    Emphatically *not* ``transient``, which is the distinction that matters.
    A rate limit or a 529 clears on its own, so refunding the attempt and
    retrying is right. A branch that conflicts with main does not clear by
    waiting: the same two trees will conflict the same way forever. Classified
    as transient it was refunded every time, so ``max_attempts`` never applied
    and the node retried without bound -- observed on a real run as 343
    failures over ten hours with ``attempts`` still reading 3.

    Retryable, because the node re-runs against a refreshed base and can
    genuinely succeed. Not escalatable, because no stronger model resolves a
    merge conflict.
    """

    retryable = True


class PatchError(ForgeError):
    """A generated patch could not be applied to the working tree."""

    retryable = True
    escalatable = True


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class GateError(ForgeError):
    """A validation gate could not run (distinct from a gate that *failed*)."""

    retryable = True
    transient = True


class ValidationFailed(ForgeError):
    """One or more gates returned a failing verdict."""

    retryable = True
    escalatable = True


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------


class Abort(ForgeError):
    """Operator-initiated shutdown. Unwinds cleanly and checkpoints."""


class HumanInputRequired(ForgeError):
    """Forge has exhausted autonomous options and needs a decision.

    Raising this is a last resort; the whole point of the platform is to avoid
    it. When raised, the orchestrator parks the node in ``blocked`` and keeps
    working on everything else in the graph.
    """
