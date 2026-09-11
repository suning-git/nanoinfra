"""
metrics.py — the training run leaves a file, without a wandb account and without
a core change.

THE PROBLEM. core prints metrics and optionally ships them to wandb; it writes
nothing to disk. The repo currently recovers training curves by grepping stdout
with a dozen different regexes. For a teaching project that is the wrong shape:
students should not need a wandb account to see their own loss curve, and the web
should not have to parse a terminal.

THE SEAM. Trainer sends BOTH the periodic training metrics and every evaluator's
results through one object — the return value of `_init_metrics()`, called on
rank 0 only — and that object is duck-typed (core's own DummyWandb is the no-op
version). So a run that writes jsonl is a wandb-shaped object plus a four-line
Trainer subclass. Overriding `_init_metrics` is the sanctioned door, not a pry bar.

Nothing here belongs in core yet: it has one consumer. When it has two it can earn
its way in (core/vision.md, "When Patterns Emerge").

    <ckpt_dir>/metrics.jsonl, one JSON object per line:
    {"step": 1200, "train/loss": 4.31, "train/mfu": 41.2, "val/motion_ce": 5.02}
"""

import json
import os

from core.training.trainer import Trainer


class JsonlRun:
    """wandb-shaped sink: .log(dict) / .finish(). Wraps whatever core would have
    returned, so turning wandb on does not turn the file off.

    APPENDS. A resumed run continues the same file rather than truncating its own
    history — the trajectory is the thing that has saved investigations here, and
    a resume that erases it is a silent loss.
    """

    def __init__(self, path, inner=None):
        self.path = str(path)
        self.inner = inner
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)   # line-buffered: readable live

    def log(self, data):
        if self.inner is not None:
            self.inner.log(data)
        self._fh.write(json.dumps({k: _jsonable(v) for k, v in data.items()}) + "\n")

    def finish(self, *a, **kw):
        try:
            self._fh.close()
        finally:
            if self.inner is not None:
                self.inner.finish(*a, **kw)

    def __getattr__(self, name):
        """Anything else the Trainer might call goes to the wrapped run (or is a
        no-op when there is none)."""
        inner = self.__dict__.get("inner")
        if inner is not None:
            return getattr(inner, name)
        return lambda *a, **kw: None


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return float(v) if hasattr(v, "__float__") else str(v)


class TrainerWithJsonl(Trainer):
    """core's Trainer, plus a local metrics file. The ONLY difference."""

    def __init__(self, *args, metrics_path=None, **kwargs):
        self.metrics_path = metrics_path
        super().__init__(*args, **kwargs)

    def _init_metrics(self):
        run = super()._init_metrics()
        if not self.metrics_path:
            return run
        from core.utils import DummyWandb
        inner = None if isinstance(run, DummyWandb) else run
        print(f"metrics -> {self.metrics_path}")
        return JsonlRun(self.metrics_path, inner=inner)
