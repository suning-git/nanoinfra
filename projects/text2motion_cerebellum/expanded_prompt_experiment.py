"""Run the preregistered OMG prompt expansion and evaluate three frozen policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


OMG_COMMIT = "61e196010b332acbace223b2c449e25f454ea0a3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log, env: dict[str, str]) -> bool:
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    return completed.returncode == 0


def checked(command: list[str], log, env: dict[str, str], label: str) -> None:
    if not run(command, log, env):
        raise RuntimeError(f"{label}_failed")


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "text2motion-expanded-prompts-preregistered-v1":
        raise ValueError("unexpected prompt protocol schema")
    prompts = payload.get("prompts", [])
    if len(prompts) != 12 or len({item["tag"] for item in prompts}) != 12:
        raise ValueError("expected 12 unique prompt tags")
    if sum(item["source"] == "new_generation" for item in prompts) != 9:
        raise ValueError("expected nine new prompts")
    return payload


def make_seed(omg_root: Path, path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(
        str(omg_root / "assets/holomotion/g1_29dof/scene_29dof.xml")
    )
    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], 120, axis=0)
    if qpos.shape != (120, 36):
        raise RuntimeError(f"unexpected seed shape {qpos.shape}")
    np.savez_compressed(path, qpos_36=qpos, fps=np.float32(30.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--omg-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tracker-repo", type=Path, required=True)
    parser.add_argument("--existing-refs", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--policy", action="append", nargs=2, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    out = args.out
    generated_root = out / "generated"
    candidate_root = out / "candidate_refs"
    accepted_root = out / "accepted_refs"
    for path in (generated_root, candidate_root, accepted_root, out / "logs"):
        path.mkdir(parents=True, exist_ok=True)

    common_env = os.environ.copy()
    common_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMG_MODELS_ROOT": str(args.model_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MUJOCO_GL": "glfw",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    omg_env = dict(common_env)
    omg_env["PYTHONPATH"] = f"{args.omg_root / 'src'}:{args.tracker_repo}:{args.root}"
    tracker_env = dict(common_env)
    tracker_env["PYTHONPATH"] = (
        f"{args.root}:{args.tracker_repo}:{args.root / 'projects/motion_cerebellum_remote'}"
    )

    seed = out / "seed_motion.npz"
    make_seed(args.omg_root, seed)
    onnx = args.model_root / "generation/onnx/50m/last_denoiser_step.onnx"
    inherited = {
        "walk_forward": args.existing_refs / "shard_000.npz",
        "walk_turn_left": args.existing_refs / "shard_001.npz",
        "walk_turn_right": args.existing_refs / "shard_002.npz",
    }
    records = []
    accepted_index = 0
    with (out / "logs/run.log").open("w", encoding="utf-8") as log:
        for item in protocol["prompts"]:
            tag = item["tag"]
            prompt = item["text"]
            source = item["source"]
            generation = "not_run"
            quality_gate = "not_run"
            candidate = candidate_root / f"{tag}.npz"
            if source == "frozen_existing":
                if tag not in inherited:
                    raise RuntimeError(f"unexpected frozen prompt {tag}")
                shutil.copy2(inherited[tag], candidate)
                generation = "inherited"
                quality_gate = "passed"
            else:
                reference = generated_root / tag / "reference_motion.npz"
                generation_command = [
                    sys.executable,
                    "-m",
                    "omg.cli.pipeline.main",
                    "--mode",
                    "diffusion-only",
                    "--diffusion-onnx",
                    str(onnx),
                    "--seed-motion",
                    str(seed),
                    "--text",
                    prompt,
                    "--num-frames",
                    "120",
                    "--providers",
                    "CUDAExecutionProvider",
                    "--torch-device",
                    "cuda",
                    "--no-compile-history-encoder",
                    "--output-root",
                    str(generated_root),
                    "--output-name",
                    tag,
                ]
                if run(generation_command, log, omg_env):
                    generation = "passed"
                    adapter_env = dict(common_env)
                    adapter_env["PYTHONPATH"] = f"{args.root}:{args.tracker_repo}"
                    adapter_command = [
                        sys.executable,
                        "-m",
                        "projects.text2motion_cerebellum.omg_adapter",
                        "convert",
                        str(reference),
                        str(candidate),
                        "--tracker-repo",
                        str(args.tracker_repo),
                        "--caption",
                        prompt,
                        "--clip",
                        tag,
                    ]
                    quality_gate = (
                        "passed" if run(adapter_command, log, adapter_env) else "rejected"
                    )
                else:
                    generation = "failed"
            accepted_shard = None
            reference_sha = None
            if quality_gate == "passed":
                accepted_shard = f"shard_{accepted_index:03d}.npz"
                accepted_path = accepted_root / accepted_shard
                shutil.copy2(candidate, accepted_path)
                reference_sha = sha256(accepted_path)
                accepted_index += 1
            records.append(
                {
                    **item,
                    "generation": generation,
                    "quality_gate": quality_gate,
                    "accepted_shard": accepted_shard,
                    "reference_sha256": reference_sha,
                }
            )

        prompt_status = {
            "schema": "text2motion-expanded-prompt-generation-v1",
            "protocol_sha256": sha256(args.protocol),
            "selection_rule": protocol["selection_rule"],
            "omg_commit": OMG_COMMIT,
            "prompts": records,
        }
        (out / "prompt_status.json").write_text(
            json.dumps(prompt_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if accepted_index < 3:
            raise RuntimeError("accepted_inventory_too_small")

        policies = [(int(seed_text), Path(path)) for seed_text, path in args.policy]
        if [seed_value for seed_value, _ in policies] != [0, 1, 2]:
            raise ValueError("policies must be supplied in training-seed order 0, 1, 2")
        for training_seed, policy in policies:
            episodes = out / f"seed{training_seed}_episodes.json"
            command = [
                str(args.train_python),
                str(args.wrapper),
                "evaluate",
                "--tracker-repo",
                str(args.tracker_repo),
                "--preview-offsets",
                "auto",
                "--",
                "--robot",
                "g1",
                "--policy",
                str(policy),
                "--label",
                f"expanded_prompts_seed{training_seed}",
                "--ref-dir",
                str(accepted_root),
                "--split",
                "all",
                "--amass",
                str(accepted_index),
                "--obs-noise",
                "--repeats",
                "4",
                "--episodes-json",
                str(episodes),
            ]
            checked(command, log, tracker_env, f"evaluate_seed{training_seed}")

        review_command = [
            str(args.train_python),
            str(args.review),
            "--protocol",
            str(args.protocol),
            "--generation",
            str(out / "prompt_status.json"),
        ]
        for training_seed, _ in policies:
            review_command.extend(
                ["--run", str(training_seed), str(out / f"seed{training_seed}_episodes.json")]
            )
        review_command.extend(["--out", str(out / "summary.json")])
        checked(review_command, log, tracker_env, "review")


if __name__ == "__main__":
    main()
