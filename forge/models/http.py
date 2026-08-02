"""A small, robust JSON-over-HTTP client built on the standard library.

Forge's model calls need exactly four things: POST JSON, read JSON, respect
timeouts, and retry the right failures with jittered backoff. ``urllib`` does
the first three and this module adds the fourth, which keeps the platform
installable with ``pip install -e .`` on a fresh box with no network wheel
fetching -- a property that matters when the platform is expected to be
resurrected on a new machine years from now.

Retry classification maps HTTP status onto Forge's error taxonomy so the layers
above can act on ``retryable``/``transient`` without parsing status codes.
"""

from __future__ import annotations

import gzip
import http.client
import json
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

from ..errors import ModelError, ProviderUnavailable, RateLimited
from ..obs.log import get_logger

log = get_logger("models.http")

#: Statuses worth retrying. 408/409 are included because some gateways use them
#: for transient upstream contention.
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: TCP keepalive: start probing after this much silence, then every
#: ``_KEEPALIVE_INTERVAL`` seconds, giving up after ``_KEEPALIVE_PROBES``.
#:
#: This is the only thing that detects a peer that went away without closing the
#: connection -- a model server restarted mid-generation, a VPN that dropped the
#: flow, a laptop that slept. Without it the socket sits ESTABLISHED and readable
#: forever, and the only limit is the read timeout, which for a long local
#: generation is measured in tens of minutes. Observed in production: a node held
#: a slot for 45 minutes against an idle server that had already forgotten the
#: request.
_KEEPALIVE_IDLE = 60
_KEEPALIVE_INTERVAL = 10
_KEEPALIVE_PROBES = 5


def _enable_keepalive(sock: socket.socket) -> None:
    """Best effort: every option here is Linux-specific and optional."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (
            ("TCP_KEEPIDLE", _KEEPALIVE_IDLE),
            ("TCP_KEEPINTVL", _KEEPALIVE_INTERVAL),
            ("TCP_KEEPCNT", _KEEPALIVE_PROBES),
        ):
            option = getattr(socket, name, None)
            if option is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
    except OSError:  # pragma: no cover - platform without these options
        pass


class _KeepAliveHTTPConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        super().connect()
        _enable_keepalive(self.sock)


class _KeepAliveHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        super().connect()
        _enable_keepalive(self.sock)


class _KeepAliveHTTPSHandler(urllib.request.HTTPSHandler):
    """Hands out keepalive-enabled connections. Replaces urllib's own handler.

    ``build_opener`` substitutes a handler for the default of the same base
    class, which is why this subclasses ``HTTPSHandler`` rather than wrapping it.
    """

    def https_open(self, req: Any) -> Any:
        # Only `context`: `check_hostname` is folded into the context by
        # HTTPSHandler.__init__ and the attribute no longer exists on modern
        # Pythons. Passing it raised AttributeError on the first real HTTPS call
        # -- which no test caught, because they all went over plain HTTP.
        # `_context` is set by HTTPSHandler.__init__ but is absent from typeshed.
        return self.do_open(_KeepAliveHTTPSConnection, req, context=self._context)  # type: ignore[attr-defined]


class _KeepAliveHTTPHandler(urllib.request.HTTPHandler):
    """The plain-HTTP twin, for a model server on the local network."""

    def http_open(self, req: Any) -> Any:
        return self.do_open(_KeepAliveHTTPConnection, req)


class HttpClient:
    def __init__(
        self,
        *,
        timeout: float = 600.0,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        backoff_max: float = 60.0,
        verify_tls: bool = True,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._ctx = ssl.create_default_context()
        if not verify_tls:  # pragma: no cover - operator override for local proxies
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE
        self._opener = urllib.request.build_opener(
            _KeepAliveHTTPSHandler(context=self._ctx), _KeepAliveHTTPHandler()
        )

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST a JSON body and return the decoded JSON response.

        Retries only on classified-transient failures. A 400 from a provider is
        a bug in the request that will fail identically forever, so it is raised
        immediately rather than burning four attempts and forty seconds.

        ``timeout`` bounds the **whole call**, not each attempt. The obvious
        reading -- socket timeout per attempt, times ``max_retries + 1``, plus
        backoff -- turns a 30-minute configured timeout into two and a half hours
        of a worker slot held by a request nobody is answering. The caller sets a
        timeout because that is how long it is willing to wait; retries are an
        implementation detail and must fit inside it.
        """
        body = json.dumps(payload).encode("utf-8")
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": "forge/0.1",
            **(headers or {}),
        }
        total_timeout = timeout or self.timeout
        deadline = time.monotonic() + total_timeout
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise last_error or ProviderUnavailable(
                    f"{_host(url)} did not answer within {total_timeout:.0f}s"
                )
            request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
            try:
                with self._opener.open(request, timeout=remaining) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                status = exc.code
                if status == 429:
                    retry_after = _retry_after(exc)
                    last_error = RateLimited(
                        f"rate limited by {_host(url)}", retry_after=retry_after, detail=detail
                    )
                    if attempt < self.max_retries:
                        self._sleep(attempt, retry_after, deadline)
                        continue
                    # `from exc` keeps the HTTP error in the traceback; without
                    # it the underlying status is lost from the crash report.
                    raise last_error from exc
                if status in _RETRY_STATUS:
                    last_error = ProviderUnavailable(
                        f"{_host(url)} returned {status}", status=status, detail=detail
                    )
                    if attempt < self.max_retries:
                        self._sleep(attempt, _retry_after(exc), deadline)
                        continue
                    raise last_error from exc
                raise ModelError(
                    f"{_host(url)} rejected request with {status}", status=status, detail=detail
                ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = ProviderUnavailable(f"cannot reach {_host(url)}: {exc}")
                if attempt < self.max_retries:
                    self._sleep(attempt, None, deadline)
                    continue
                raise last_error from exc
            except json.JSONDecodeError as exc:
                raise ModelError(f"{_host(url)} returned non-JSON body: {exc}") from exc

        raise last_error or ModelError("request failed")  # pragma: no cover

    def _sleep(self, attempt: int, retry_after: float | None, deadline: float | None = None) -> None:
        if retry_after is not None:
            delay = min(retry_after, self.backoff_max)
        else:
            delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        # Full jitter. Several workers hitting the same rate-limited endpoint
        # must not resynchronise into a thundering herd.
        delay = random.uniform(0, delay)
        if deadline is not None:
            # Never sleep past the caller's deadline: waiting out a backoff only
            # to abandon the request is pure latency.
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        log.debug("retrying after backoff", attempt=attempt + 1, delay=round(delay, 2))
        time.sleep(delay)


def _host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc or url
    except Exception:  # pragma: no cover
        return url


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
        if exc.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", "replace")
    except Exception:  # pragma: no cover
        return ""
    return text[:2000]


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    if not value:
        return None
    try:
        return float(value)
    except ValueError:  # pragma: no cover - HTTP-date form
        return None
