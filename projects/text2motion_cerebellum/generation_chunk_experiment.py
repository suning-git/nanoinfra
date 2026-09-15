"""Compare documented one- and two-chunk OMG generation on frozen prompts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


VARIANTS = (
    {"name": "documented_single_chunk_60", "frames": 60, "repeat": 1},
    {"name": "documented_two_chunks_120", "frames": 120, "repeat": 2},
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-protocol", type=Path, required=True)
    parser.add_argument("--omg-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--seed-motion", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.prompt_protocol.read_text(encoding="utf-8"))
    prompts = [row for row in protocol["prompts"] if row["source"] == "new_generation"]
    if len(prompts) != 9:
        raise ValueError("expected nine frozen new-generation prompts")
    onnx = args.model_root / "generation/onnx/50m/last_denoiser_step.onnx"
    if not onnx.is_file() or not args.seed_motion.is_file():
        raise FileNotFoundError("generation artifact or seed motion missing")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{args.omg_root / 'src'}:{args.tracker_repo}:{args.root}",
            "OMG_MODELS_ROOT": str(args.model_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    generation_environment = dict(environment)
    generation_environment["MUJOCO_GL"] = "glfw"
    probe_environment = dict(environment)
    probe_environment.pop("MUJOCO_GL", None)
    records = []
    for variant in VARIANTS:
        for prompt in prompts:
            tag = prompt["tag"]
            output_name = f"{variant['name']}__{tag}"
            generated_root = args.out / "generated"
            reference = generated_root / output_name / "reference_motion.npz"
            condition = (
                f"text: {prompt['text']}"
                if variant["repeat"] == 1
                else f"text[{variant['repeat']}]: {prompt['text']}"
            )
            generation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "omg.cli.pipeline.main",
                    "--mode",
                    "diffusion-only",
                    "--diffusion-onnx",
                    str(onnx),
                    "--seed-motion",
                    str(args.seed_motion),
                    "--condition-sequence",
                    condition,
                    "--num-frames",
                    str(variant["frames"]),
                    "--providers",
                    "CUDAExecutionProvider",
                    "--torch-device",
                    "cuda",
                    "--no-compile-history-encoder",
                    "--output-root",
                    str(generated_root),
                    "--output-name",
                    output_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=generation_environment,
                text=True,
                check=False,
            )
            row = {
                "variant": variant["name"],
                "requested_frames": variant["frames"],
                "tag": tag,
                "text": prompt["text"],
                "generation": "passed" if generation.returncode == 0 else "failed",
            }
            if generation.returncode != 0:
                error_lines = [
                    line.strip() for line in generation.stderr.splitlines() if line.strip()
                ]
                row["generation_error_summary"] = (
                    error_lines[-1][:300] if error_lines else "no_stderr"
                )
            if generation.returncode == 0 and reference.is_file():
                probe_out = args.out / "probes" / f"{output_name}.json"
                probe = subprocess.run(
                    [
                        sys.executable,
                        str(args.probe_script),
                        "--source",
                        str(reference),
                        "--tracker-repo",
                        str(args.tracker_repo),
                        "--tag",
                        tag,
                        "--out",
                        str(probe_out),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=probe_environment,
                    check=False,
                )
                if probe.returncode == 0:
                    record = json.loads(probe_out.read_text(encoding="utf-8"))["record"]
                    row.update(
                        {
                            "quality_gate": record["upstream_gate"]["result"],
                            "gate_reason": record["upstream_gate"]["reason"],
                            "raw_omg_30hz": record["raw_omg_30hz"],
                            "bridge_50hz": record["bridge_50hz"],
                        }
                    )
                else:
                    row["quality_gate"] = "probe_failed"
            else:
                row["quality_gate"] = "not_run"
            records.append(row)
            write(
                args.out / "result.json",
                {
                    "schema": "text2motion-generation-chunk-experiment-v1",
                    "result": "running",
                    "selection_rule": "one generation attempt per frozen prompt and variant; no rerolls",
                    "records": records,
                },
            )

    summaries = {}
    for variant in VARIANTS:
        rows = [row for row in records if row["variant"] == variant["name"]]
        reasons = Counter(row.get("gate_reason") for row in rows if row.get("quality_gate") == "rejected")
        summaries[variant["name"]] = {
            "generated": sum(row["generation"] == "passed" for row in rows),
            "quality_passed": sum(row.get("quality_gate") == "passed" for row in rows),
            "quality_rejected": sum(row.get("quality_gate") == "rejected" for row in rows),
            "reason_counts": dict(sorted(reasons.items())),
            "midpoint_worst_step_count": sum(
                row.get("raw_omg_30hz", {}).get("joint_step_worst_transition") == [59, 60]
                for row in rows
            ),
        }
    write(
        args.out / "result.json",
        {
            "schema": "text2motion-generation-chunk-experiment-v1",
            "result": (
                "passed"
                if all(row["generation"] == "passed" for row in records)
                else "invalid_generation_preflight"
            ),
            "selection_rule": "one generation attempt per frozen prompt and variant; no rerolls",
            "variants": list(VARIANTS),
            "summaries": summaries,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
