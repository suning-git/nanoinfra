"""The examples in core's documentation run, and print what the documentation says they print.

Every .md under core/ is read, and its fenced code blocks are taken by their info string:

  ```python run      executed on the CPU; the run blocks of one doc go in order, in one namespace
  ```text output     right after a run block: its expected stdout, compared line by line (trailing spaces ignored)
  ```python gpu      needs a GPU: executed only when CUDA is available, each in a fresh interpreter (it may set
                     process-wide things, as set_torch_global_defaults does, which a test process that has already
                     used CUDA would refuse), so it must stand alone
  ```python          an illustration: not executed

So a change to core that makes an example wrong fails here, and the doc is fixed with the code. Examples write only
under /tmp.
"""
import contextlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CORE = Path(__file__).resolve().parents[2]
REPO = CORE.parent
DOCS = sorted(CORE.rglob("*.md"))
FENCE = re.compile(r"^```(.*)$")


def blocks(text):
    """[(info, body, line)] of the fenced blocks, in order; `line` is the opening fence's line number."""
    out, info, body, start = [], None, [], 0
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if info is None and m:
            info, body, start = m.group(1).strip(), [], n
        elif info is not None and line.strip() == "```":
            out.append((info, "\n".join(body), start))
            info = None
        elif info is not None:
            body.append(line)
    assert info is None, f"unclosed code fence opened at line {start}"
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(CORE)))
def test_doc_examples(doc):
    rel = doc.relative_to(REPO)
    found = blocks(doc.read_text(encoding="utf-8"))
    namespace = {"__name__": f"doc:{rel}"}
    for i, (info, body, line) in enumerate(found):
        words = info.split()
        if words[:1] != ["python"]:
            continue
        if "gpu" in words[1:]:
            if torch.cuda.is_available():
                done = subprocess.run([sys.executable, "-c", body], cwd=REPO, capture_output=True, text=True,
                                      env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                assert done.returncode == 0, f"{rel}:{line}: the GPU example failed:\n{done.stderr[-3000:]}"
            continue
        if "run" not in words[1:]:
            continue
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(body, f"{rel}:{line}", "exec"), namespace)
        nxt = found[i + 1] if i + 1 < len(found) else None
        if nxt is not None and nxt[0].split() == ["text", "output"]:
            got = [s.rstrip() for s in stdout.getvalue().rstrip("\n").splitlines()]
            want = [s.rstrip() for s in nxt[1].rstrip("\n").splitlines()]
            assert got == want, (f"{rel}:{line}: the example's output differs from the block at line {nxt[2]}\n"
                                 f"expected:\n" + "\n".join(want) + "\ngot:\n" + "\n".join(got))
