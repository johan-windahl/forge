"""Built-in gates.

Importing this package registers every gate. New gates are added by dropping a
module here and importing it below, or by calling
``forge.validation.gate.register`` from anywhere -- including a project-local
plugin, which is how a project adds a check Forge does not ship.
"""

from . import browser, command, perf, security, visual

__all__ = ["browser", "command", "perf", "security", "visual"]
