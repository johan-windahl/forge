"""Cheap reachability probes for model providers.

A run that cannot reach its model server does not fail: it waits. The inner
executor blocks on a socket, the node holds its lease, and the only outward sign
is that the usage report goes quiet. That silence is indistinguishable from a
model thinking hard, which is how a broken endpoint cost one run nearly two
hours before anyone looked.

These probes cost nothing -- an unauthenticated GET against ``/models`` -- and
exist so both ``forge doctor`` and the running orchestrator can turn that
silence into a named error.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from ..config import Config, ProviderConfig

#: Deliberately short. This is a liveness question, not a request: a server that
#: cannot answer ``/models`` in this long is not one the run should be waiting on.
PROBE_TIMEOUT = 5.0


def probe_provider(provider: ProviderConfig, *, timeout: float = PROBE_TIMEOUT) -> tuple[bool, str]:
    """Ask an OpenAI-compatible provider whether it is answering.

    Returns ``(reachable, explanation)``. An HTTP error status still counts as
    reachable: the server answered, which is what the caller needs to know. Only
    a transport failure means "not there".
    """
    url = provider.base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500, f"{provider.base_url} responded {response.status}"
    except urllib.error.HTTPError as exc:
        return True, f"{provider.base_url} responded {exc.code}"
    except Exception as exc:
        return False, f"{provider.base_url} unreachable: {exc}"


def probe_local(config: Config, *, timeout: float = PROBE_TIMEOUT) -> tuple[bool, str]:
    """Probe the ``local`` provider, if one is configured."""
    provider = config.models.providers.get("local")
    if provider is None:
        return False, "no local provider configured"
    return probe_provider(provider, timeout=timeout)
