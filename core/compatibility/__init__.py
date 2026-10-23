"""Helpers that let the current core read what an older core wrote.

Rules for this package:

  - Runtime code never calls these helpers. Where it refuses an old format (today: MixedDataLoader.set_state, for
    the list snapshot), the error names the helper to use; an orchestrator or a user calls the helper explicitly, so
    reading an old format is a visible decision, not a silent fallback.
  - Saved artifacts stay as they are by default: the first choice is a helper that converts at load time, so the old
    code can still read the same checkpoint. A one-time rewrite (meta.json only, with a backup) is offered as well.
  - One file per format change. Its docstring says which format changed, since when, which old formats it accepts,
    what it writes, and when the file can be deleted.
  - Call sites write `from core import compatibility; compatibility.accept_list_snapshot(...)`, so it shows there
    that the call exists for compatibility; function names say which old format they handle.

Contents:
  loader_snapshot.py   the data snapshot went from a list of source states (matched by position) to a dict keyed
                       by source name: accept_list_snapshot (wrap a loader at load time), upgrade_list_snapshot
                       (rewrite a checkpoint's meta.json once; also `python -m core.compatibility.loader_snapshot`)
"""

import importlib

_EXPORTS = {"accept_list_snapshot": "loader_snapshot", "upgrade_list_snapshot": "loader_snapshot"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # Imported on first use rather than at the top of this file: `python -m core.compatibility.loader_snapshot`
    # then runs that module once; importing it here first would make runpy warn that it is about to run it again.
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
