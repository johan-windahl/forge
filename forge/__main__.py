"""``python -m forge``.

Exists so a detached run can be respawned as ``sys.executable -m forge`` when the
console script is not on PATH -- inside a venv invoked by absolute path, or from
a checkout run with ``python -m``. See ``forge.kernel.daemon``.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
