"""The HTTP client's two hard guarantees: a bounded wait, and noticing a dead peer.

Both are tested against a real socket. The failure that motivated them could not
have been caught with a mocked transport: a worker held a slot for 45 minutes on
a connection that was ESTABLISHED to a server which had already forgotten the
request, because the configured timeout was per attempt and there were five
attempts.
"""

from __future__ import annotations

import http.server
import json
import socket
import socketserver
import threading
import time
from typing import Any

import pytest

from forge.errors import ModelError, ProviderUnavailable
from forge.models.http import HttpClient, _KeepAliveHTTPConnection


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _serve(handler_cls: type) -> tuple[str, _Server]:
    server = _Server(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/v1/chat", server


def _handler(status: int, body: dict[str, Any] | None = None, *, delay: float = 0.0) -> type:
    class H(http.server.BaseHTTPRequestHandler):
        attempts = 0

        def do_POST(self) -> None:
            H.attempts += 1
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if delay:
                time.sleep(delay)
            payload = json.dumps(body or {"ok": True}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    return H


def test_a_successful_post_returns_the_decoded_body() -> None:
    url, server = _serve(_handler(200, {"choices": [{"text": "hi"}]}))
    try:
        assert HttpClient(timeout=5).post_json(url, {"q": 1})["choices"][0]["text"] == "hi"
    finally:
        server.shutdown()


def test_the_timeout_bounds_the_whole_call_not_each_attempt() -> None:
    """The bug this exists to prevent.

    With a per-attempt timeout and five attempts, a 30-minute configured wait
    became two and a half hours of a held worker slot. A caller sets a timeout
    because that is how long it is willing to wait; retries have to fit inside it.
    """
    handler = _handler(503)
    url, server = _serve(handler)
    client = HttpClient(timeout=1.0, max_retries=4, backoff_base=30.0, backoff_max=60.0)
    try:
        started = time.monotonic()
        with pytest.raises(ProviderUnavailable):
            client.post_json(url, {"q": 1})
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
    # Without the deadline this is minutes: four backoffs of up to 60s each.
    assert elapsed < 3.0, f"took {elapsed:.1f}s, so the deadline is not being enforced"


def test_backoff_never_sleeps_past_the_deadline() -> None:
    """Waiting out a backoff only to abandon the request is pure latency."""
    client = HttpClient(timeout=10.0, backoff_base=30.0, backoff_max=60.0)
    started = time.monotonic()
    client._sleep(3, None, deadline=time.monotonic() + 0.2)
    assert time.monotonic() - started < 1.0


def test_retries_still_happen_inside_the_deadline() -> None:
    """The bound must not have turned retrying off."""
    handler = _handler(503)
    url, server = _serve(handler)
    client = HttpClient(timeout=5.0, max_retries=3, backoff_base=0.01, backoff_max=0.05)
    try:
        with pytest.raises(ProviderUnavailable):
            client.post_json(url, {"q": 1})
    finally:
        server.shutdown()
    assert handler.attempts == 4  # the first plus three retries


def test_a_permanent_rejection_is_not_retried() -> None:
    handler = _handler(400)
    url, server = _serve(handler)
    try:
        with pytest.raises(ModelError):
            HttpClient(timeout=5).post_json(url, {"q": 1})
    finally:
        server.shutdown()
    assert handler.attempts == 1


def test_connections_enable_tcp_keepalive() -> None:
    """The only thing that detects a peer that vanished without closing.

    A model server restarted mid-generation, a VPN that dropped the flow: the
    socket stays ESTABLISHED and the read timeout -- tens of minutes for a long
    local generation -- is otherwise the only limit.
    """
    url, server = _serve(_handler(200))
    port = server.server_address[1]
    try:
        conn = _KeepAliveHTTPConnection("127.0.0.1", port, timeout=5)
        conn.connect()
        try:
            assert conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
            idle = getattr(socket, "TCP_KEEPIDLE", None)
            if idle is not None:
                assert conn.sock.getsockopt(socket.IPPROTO_TCP, idle) == 60
        finally:
            conn.close()
    finally:
        server.shutdown()
    assert url  # the server really was listening


def test_the_client_uses_keepalive_connections() -> None:
    """Belt and braces: the opener has to be wired to the handlers above.

    Enabling keepalive on a connection class nobody uses is the easiest way for
    this fix to be silently undone.
    """
    client = HttpClient(timeout=5)
    classes = set()
    for handler in client._opener.handlers:
        for attr in ("http_open", "https_open"):
            if hasattr(handler, attr):
                classes.add(type(handler).__name__)
    assert "_KeepAliveHTTPHandler" in classes
    assert "_KeepAliveHTTPSHandler" in classes


def test_the_https_path_is_exercised_not_just_the_http_one() -> None:
    """A handler that raises on its first real HTTPS call is worse than no handler.

    The first attempt at this passed `check_hostname` to `do_open`; the attribute
    no longer exists, so every HTTPS model call died with an AttributeError. Every
    test above goes over plain HTTP and none of them noticed. This one drives
    `https_open` far enough to hit any such mistake -- a refused connection is a
    URLError, and anything else is a bug in the handler itself.
    """
    import urllib.error

    client = HttpClient(timeout=1, max_retries=0)
    # Port 1 is reserved and never listening: the call gets as far as connect().
    with pytest.raises((ProviderUnavailable, urllib.error.URLError)):
        client.post_json("https://127.0.0.1:1/v1/chat", {"q": 1})
