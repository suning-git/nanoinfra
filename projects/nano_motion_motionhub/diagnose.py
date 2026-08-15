"""Emit a redacted, structured summary of a remote task log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SECRET = re.compile(r"(?:hf_|token[=: ]+)[A-Za-z0-9_-]{8,}", re.IGNORECASE)
EXCEPTION = re.compile(
    r"^(?:\[[^]]+\]:\s*)?"
    r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit)):\s*(.*)$"
)
SIGNAL = re.compile(
    r"traceback|error|failed|invalid|out of memory|assert|exception",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    lines = args.log.read_text(errors="replace").splitlines()
    exceptions = []
    signals = []
    for line in lines:
        candidate = line.strip()
        if SIGNAL.search(candidate) and "warning" not in candidate.lower():
            candidate = SECRET.sub("[REDACTED]", candidate)[:240]
            if candidate and candidate not in signals:
                signals.append(candidate)
        match = EXCEPTION.match(line.strip())
        if match:
            kind, detail = match.groups()
            detail = SECRET.sub("[REDACTED]", detail)[:240]
            item = {"type": kind, "message": detail}
            if item not in exceptions:
                exceptions.append(item)
    exception_type = "unknown"
    message = "no exception line found"
    if exceptions:
        exception_type = exceptions[-1]["type"]
        message = exceptions[-1]["message"]
    if exception_type == "unknown":
        # ``SystemExit(<message>)`` is printed by Python as a plain final line,
        # without the exception class.  Keep the remote log private but retain a
        # short, redacted terminal message in the structured diagnostic.
        for line in reversed(lines):
            candidate = line.strip()
            if candidate:
                exception_type = "SystemExitOrShellError"
                message = candidate
                break
    message = SECRET.sub("[REDACTED]", message)[:240]
    outputs = {}
    if args.out_dir.exists():
        for name in (
            "pilot_prepare.json",
            "pilot_training.json",
            "formal_prepare.json",
            "formal_training.json",
            "demo.tar.gz",
        ):
            path = args.out_dir / name
            outputs[name] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
    report = {
        "schema": "nano-motion-motionhub-diagnostic-v1",
        "exception_type": exception_type,
        "message": message,
        "exceptions": exceptions[-5:],
        "signals": signals[-8:],
        "log_lines": len(lines),
        "outputs": outputs,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
