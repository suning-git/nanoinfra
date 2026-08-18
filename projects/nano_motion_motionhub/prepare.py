"""Build nano_motion Text2Motion caches from MotionHub HumanML3D-AMASS.

MotionHub publishes HumanML3D clips as 30 fps, Y-up SMPL-H parameters.  The
course rot139 converter consumes Z-up AMASS parameters.  This adapter preserves
the SMPL translation semantics while rotating the whole posed body into Z-up,
then uses the existing BONES-SEED codec to produce the standard
``t2m_motionhub_{train,val}.npz`` cache contract.

The original HumanML3D captions and split files come from the course starter.
MotionHub provides the already-cropped motion with the same clip identifier, so
no frame-range reconstruction or approximate caption pairing is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ID = "ZeyuLing/MotionHub"
MOTION_PREFIX = "HumanML3D_AMASS/smplh_52"
MAX_CAPTIONS = 4
MIN_CODES = 6
TARGET_FPS = 30.0
Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def captions_for(starter: Path, clip_id: str) -> list[str]:
    path = starter / "humanml3d" / "texts" / f"{clip_id}.txt"
    if not path.is_file():
        return []
    captions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        caption = line.split("#", 1)[0].strip()
        if caption and caption not in captions:
            captions.append(caption)
        if len(captions) >= MAX_CAPTIONS:
            break
    return captions


def split_ids(starter: Path, split: str) -> list[str]:
    path = starter / "humanml3d" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing HumanML3D split: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def motionhub_filename(clip_id: str) -> str:
    # MotionHub shards both ordinary ids (0000/000000.npz) and mirrored ids
    # (M000/M000000.npz) by the first four characters.
    return f"{MOTION_PREFIX}/{clip_id[:4]}/{clip_id}.npz"


def available_motion_ids(
    cache_dir: Path, revision: str, target_ids: Iterable[str], workers: int
) -> tuple[set[str], set[str]]:
    """Index only the fixed-revision shards needed by the requested split.

    The complete MotionHub subtree is large enough that recursively listing it
    dominates a small training run.  Files are sharded by the first four ID
    characters, so a few small directory listings provide an exact existence
    test for the relevant part of the course split.  Results persist across
    retries and larger runs.
    """
    from huggingface_hub import HfApi

    index = cache_dir / f"motionhub_smplh52_shards_{revision[:16]}.json"
    payload = {"prefixes": [], "ids": []}
    if index.is_file():
        payload = json.loads(index.read_text())
    indexed = set(payload.get("prefixes", []))
    ids = set(payload.get("ids", []))
    wanted = sorted({clip_id[:4] for clip_id in target_ids} - indexed)

    def one(prefix: str) -> tuple[str, list[str], str | None]:
        try:
            entries = HfApi().list_repo_tree(
                REPO_ID,
                path_in_repo=f"{MOTION_PREFIX}/{prefix}",
                recursive=False,
                expand=False,
                repo_type="dataset",
                revision=revision,
            )
            found = [
                Path(getattr(entry, "path", "")).stem
                for entry in entries
                if getattr(entry, "path", "").endswith(".npz")
            ]
            return prefix, found, None
        except Exception as exc:
            return prefix, [], type(exc).__name__

    unexpected: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(max(1, workers), 16)) as pool:
        futures = {pool.submit(one, prefix): prefix for prefix in wanted}
        for future in as_completed(futures):
            prefix, found, error = future.result()
            # A missing shard is an exact negative lookup.  Network and server
            # failures must not be cached as if the source lacked the data.
            if error and "EntryNotFound" not in error:
                unexpected[prefix] = error
                continue
            indexed.add(prefix)
            ids.update(found)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"prefixes": sorted(indexed), "ids": sorted(ids)}
    temp = index.with_suffix(".tmp")
    temp.write_text(json.dumps(payload) + "\n")
    temp.replace(index)
    if unexpected:
        kinds = sorted(set(unexpected.values()))
        raise RuntimeError(f"MotionHub shard inventory failed: {kinds}")
    return ids, indexed


def download_batch(
    ids: Iterable[str], cache_dir: Path, revision: str, workers: int
) -> tuple[dict[str, Path], dict[str, str]]:
    from huggingface_hub import hf_hub_download

    def one(clip_id: str) -> tuple[str, Path]:
        # A commit-pinned snapshot path is immutable.  Reuse it directly so a
        # resumed large run does not spend one Hub metadata request per clip
        # merely to rediscover files already present on persistent storage.
        cached = (
            cache_dir
            / "datasets--ZeyuLing--MotionHub"
            / "snapshots"
            / revision
            / motionhub_filename(clip_id)
        )
        if cached.is_file():
            return clip_id, cached
        for attempt in range(4):
            try:
                path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=motionhub_filename(clip_id),
                    revision=revision,
                    cache_dir=str(cache_dir),
                )
                return clip_id, Path(path)
            except Exception:
                if attempt == 3:
                    raise
                # The mainland Hub route occasionally closes concurrent small-file
                # transfers mid-response.  Retrying at the clip boundary is safe
                # because Hub blobs and the commit-pinned snapshot are immutable.
                time.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")

    found: dict[str, Path] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, clip_id): clip_id for clip_id in ids}
        for future in as_completed(futures):
            clip_id = futures[future]
            try:
                key, path = future.result()
                found[key] = path
            except Exception as exc:  # retain a structured transient error count
                errors[clip_id] = type(exc).__name__
    return found, errors


def fetch_split(
    starter: Path,
    split: str,
    limit: int,
    cache_dir: Path,
    revision: str,
    workers: int,
    available: set[str],
    indexed_prefixes: set[str],
) -> tuple[list[tuple[str, Path, list[str]]], dict]:
    ids = split_ids(starter, split)
    candidates: list[tuple[str, list[str]]] = []
    missing_from_source = 0
    outside_inventory = 0
    missing_captions = 0
    for clip_id in ids:
        if clip_id[:4] not in indexed_prefixes:
            outside_inventory += 1
            continue
        if clip_id not in available:
            missing_from_source += 1
            continue
        captions = captions_for(starter, clip_id)
        if not captions:
            missing_captions += 1
            continue
        candidates.append((clip_id, captions))

    selected: list[tuple[str, Path, list[str]]] = []
    error_types: dict[str, int] = {}
    attempted = 0
    downloaded = 0
    chunk_size = max(32, workers * 8)
    for offset in range(0, len(candidates), chunk_size):
        if len(selected) >= limit:
            break
        batch = candidates[offset : offset + chunk_size]
        batch_ids = [item[0] for item in batch]
        captions_by_id = dict(batch)
        found, errors = download_batch(batch_ids, cache_dir, revision, workers)
        attempted += len(batch_ids)
        for kind in errors.values():
            error_types[kind] = error_types.get(kind, 0) + 1
        for clip_id in batch_ids:  # preserve official split order deterministically
            path = found.get(clip_id)
            if path is None:
                continue
            downloaded += 1
            if len(selected) < limit:
                selected.append((clip_id, path, captions_by_id[clip_id]))
    return selected, {
        "requested": limit,
        "selected": len(selected),
        "available_with_captions": len(candidates),
        "missing_from_source": missing_from_source,
        "outside_inventory": outside_inventory,
        "attempted": attempted,
        "downloaded": downloaded,
        "download_errors": sum(error_types.values()),
        "missing_captions": missing_captions,
        "downloaded_beyond_limit": max(0, downloaded - len(selected)),
        "error_types": error_types,
    }


def motionhub_to_rot139(path: Path, joints: np.ndarray, parents: np.ndarray):
    """Convert one MotionHub Y-up SMPL-H clip to course Z-up rot139.

    ``trans`` is a body-model translation parameter, not the pelvis position.
    Therefore rotating it directly would move the pelvis by the neutral model's
    non-zero root offset.  Rotate ``trans + J_root`` and subtract ``J_root`` back
    so the full posed body undergoes one rigid coordinate transform.
    """
    from modalities.motion.data import geometry as G
    from modalities.motion.data.converters import smpl_body as B
    from modalities.motion.data.converters.soma_retarget import positions_to_smpl_local

    data = np.load(path, allow_pickle=False)
    if "poses" not in data or "trans" not in data:
        raise ValueError("missing poses/trans")
    poses = np.asarray(data["poses"], dtype=np.float64)
    trans = np.asarray(data["trans"], dtype=np.float64)
    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else TARGET_FPS
    if len(poses) != len(trans) or poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(f"invalid SMPL-H arrays poses={poses.shape} trans={trans.shape}")
    if fps > TARGET_FPS + 1e-3:
        step = max(1, int(round(fps / TARGET_FPS)))
        poses, trans = poses[::step], trans[::step]
    elif fps < TARGET_FPS - 1e-3:
        raise ValueError(f"unexpected fps={fps}; refusing silent temporal upsampling")
    if len(poses) < 24:
        raise ValueError(f"clip too short: {len(poses)} frames")

    source_r = B.axis_angle_to_matrix(poses[:, :66].reshape(-1, 22, 3))
    source_r[:, 0] = np.einsum("ij,tjk->tik", Y_UP_TO_Z_UP, source_r[:, 0])
    trans = (trans + joints[0]) @ Y_UP_TO_Z_UP.T - joints[0]

    # The BONES-SEED codec was trained on soma_retarget's position-based,
    # minimal-twist rotations.  Raw SMPL fitting contains large axial twists that
    # leave joint positions unchanged but are far outside that codec's training
    # distribution.  Re-solve local rotations from the posed joint positions via
    # the exact same repository routine so the two corpora share a representation,
    # not merely the same 139-column shape.
    source_pos, _ = B.smpl_fk(source_r, trans, joints, parents)
    local_r = positions_to_smpl_local(source_pos, joints)
    trans = source_pos[:, 0] - joints[0]
    retargeted_pos, _ = B.smpl_fk(local_r, trans, joints, parents)
    foot_z = retargeted_pos[:, [10, 11], 2].min(axis=1)
    trans[:, 2] -= np.median(foot_z)

    rot6d = G.matrix_to_6d(local_r).reshape(len(local_r), -1)
    disp = np.zeros((len(local_r), 2), dtype=np.float64)
    disp[1:] = trans[1:, B.HORIZ_AXES] - trans[:-1, B.HORIZ_AXES]
    height = trans[:, B.UP_AXIS : B.UP_AXIS + 1]
    global_pos, _ = B.smpl_fk(local_r, trans, joints, parents)
    contacts = B.foot_contacts(global_pos)
    features = np.concatenate([rot6d, disp, height, contacts], axis=-1).astype(np.float32)
    if features.shape[1] != 139 or not np.isfinite(features).all():
        raise ValueError("non-finite or non-rot139 result")
    foot_floor = float(np.min(global_pos[:, [7, 10, 8, 11], 2]))
    displacement_p99 = float(np.quantile(np.linalg.norm(disp, axis=1), 0.99))
    stats = {
        "frames": int(len(features)),
        "root_height_median": float(np.median(height)),
        "foot_floor_min": foot_floor,
        "displacement_p99": displacement_p99,
        "contact_rate": float(np.mean(contacts)),
    }
    return features, stats


def root_relative_mpjpe_cm(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    from modalities.motion.data.converters import smpl_body as B
    from modalities.motion.data.converters import smpl_to_rot139 as conv

    length = min(len(reference), len(reconstruction))
    ref_r, ref_t = conv.features_to_smpl(reference[:length], np.zeros(3, np.float32))
    rec_r, rec_t = conv.features_to_smpl(reconstruction[:length], np.zeros(3, np.float32))
    ref_p, _ = B.smpl_fk(ref_r, ref_t, _MODEL_JOINTS, _MODEL_PARENTS)
    rec_p, _ = B.smpl_fk(rec_r, rec_t, _MODEL_JOINTS, _MODEL_PARENTS)
    ref_p = ref_p - ref_p[:, :1]
    rec_p = rec_p - rec_p[:, :1]
    return float(np.linalg.norm(ref_p - rec_p, axis=-1).mean() * 100.0)


_MODEL_JOINTS: np.ndarray
_MODEL_PARENTS: np.ndarray


def encode_split(records, codec, split: str, out_dir: Path, recon_samples: int):
    codes_list: list[np.ndarray] = []
    captions_list: list[list[str]] = []
    ids: list[str] = []
    clip_stats: list[dict] = []
    recon: list[float] = []
    rejected: dict[str, int] = {}
    for index, (clip_id, path, captions) in enumerate(records):
        try:
            features, stats = motionhub_to_rot139(path, _MODEL_JOINTS, _MODEL_PARENTS)
            codes = np.asarray(codec.encode(features), dtype=np.int16)
            if len(codes) < MIN_CODES:
                raise ValueError(f"only {len(codes)} codes")
            if len(recon) < recon_samples:
                decoded = np.asarray(codec.decode(codes), dtype=np.float32)
                recon.append(root_relative_mpjpe_cm(features, decoded))
        except Exception as exc:
            key = f"{type(exc).__name__}: {str(exc)[:80]}"
            rejected[key] = rejected.get(key, 0) + 1
            continue
        codes_list.append(codes)
        captions_list.append(captions)
        ids.append(clip_id)
        clip_stats.append(stats)
        if index % 100 == 0:
            print(f"{split}: encoded {len(codes_list)}/{len(records)}", flush=True)

    if not codes_list:
        raise RuntimeError(f"no usable {split} clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"t2m_motionhub_{split}.npz"
    np.savez(
        path,
        codes=np.asarray(codes_list, dtype=object),
        captions=np.asarray(captions_list, dtype=object),
        ids=np.asarray(ids),
    )
    summary = {
        "clips": len(codes_list),
        "pairs": int(sum(len(group) for group in captions_list)),
        "frames": int(sum(item["frames"] for item in clip_stats)),
        "code_length": {
            "min": int(min(map(len, codes_list))),
            "median": float(np.median(list(map(len, codes_list)))),
            "max": int(max(map(len, codes_list))),
        },
        "root_height_median": float(np.median([s["root_height_median"] for s in clip_stats])),
        "foot_floor_min_median": float(np.median([s["foot_floor_min"] for s in clip_stats])),
        "displacement_p99_median": float(np.median([s["displacement_p99"] for s in clip_stats])),
        "contact_rate_mean": float(np.mean([s["contact_rate"] for s in clip_stats])),
        "codec_rootrel_mpjpe_cm": {
            "samples": len(recon),
            "median": float(np.median(recon)) if recon else math.nan,
            "mean": float(np.mean(recon)) if recon else math.nan,
            "max": float(np.max(recon)) if recon else math.nan,
        },
        "rejected": rejected,
        "cache": path.name,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def quality_gate(report: dict) -> dict[str, bool]:
    train, val = report["splits"]["train"], report["splits"]["val"]
    return {
        "usable_train_fraction": train["clips"] >= 0.90 * report["download"]["train"]["requested"],
        "usable_val_fraction": val["clips"] >= 0.90 * report["download"]["val"]["requested"],
        "plausible_root_height": all(
            0.55 <= report["splits"][s]["root_height_median"] <= 1.35
            for s in ("train", "val")
        ),
        "floor_aligned": all(
            abs(report["splits"][s]["foot_floor_min_median"]) <= 0.15
            for s in ("train", "val")
        ),
        "no_translation_spikes": all(
            report["splits"][s]["displacement_p99_median"] <= 0.35
            for s in ("train", "val")
        ),
        "codec_domain_gate": train["codec_rootrel_mpjpe_cm"]["median"] <= 35.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--val-limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reconstruction-samples", type=int, default=16)
    parser.add_argument(
        "--cached-inventory-complete",
        action="store_true",
        help=(
            "reuse the fixed-revision shard inventory without negative directory "
            "lookups; requires an existing index large enough for both limits"
        ),
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from modalities.motion.data.converters import smpl_body as B
    from modalities.motion.tokenizers._convae import MotionCodec

    if args.train_limit < 1 or args.val_limit < 1:
        raise SystemExit("split limits must be positive")
    revision = HfApi().dataset_info(REPO_ID).sha
    inventory_targets = []
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        ids = split_ids(args.starter, split)
        scan = min(len(ids), max(limit * 2, limit + 256))
        inventory_targets.extend(ids[:scan])
    inventory_mode = "queried"
    if args.cached_inventory_complete:
        index = args.cache_dir / f"motionhub_smplh52_shards_{revision[:16]}.json"
        if not index.is_file():
            raise FileNotFoundError(f"cached inventory is missing: {index}")
        payload = json.loads(index.read_text())
        available = set(payload.get("ids", []))
        indexed_prefixes = set(payload.get("prefixes", []))
        if len(available) < args.train_limit + args.val_limit:
            raise RuntimeError(
                "cached inventory is too small for the requested split limits: "
                f"{len(available)} < {args.train_limit + args.val_limit}"
            )
        inventory_mode = "cached_fixed_revision"
    else:
        available, indexed_prefixes = available_motion_ids(
            args.cache_dir, revision, inventory_targets, args.workers
        )
    global _MODEL_JOINTS, _MODEL_PARENTS
    _MODEL_JOINTS, _MODEL_PARENTS = B.load_body_model("neutral")
    codec = MotionCodec(str(args.codec), device="cuda")

    report = {
        "schema": "nano-motion-motionhub-cache-v1",
        "source": REPO_ID,
        "source_revision": revision,
        "indexed_motion_files": len(available),
        "indexed_shards": len(indexed_prefixes),
        "inventory_mode": inventory_mode,
        "motion_subset": "HumanML3D_AMASS/smplh_52",
        "caption_source": "HumanML3D course starter",
        "coordinate_conversion": (
            "MotionHub Y-up to course Z-up, then BONES position-based minimal-twist retarget"
        ),
        "codec": args.codec.name,
        "download": {},
        "splits": {},
    }
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        records, download = fetch_split(
            args.starter,
            split,
            limit,
            args.cache_dir,
            revision,
            args.workers,
            available,
            indexed_prefixes,
        )
        report["download"][split] = download
        report["splits"][split] = encode_split(
            records, codec, split, args.out_dir, args.reconstruction_samples
        )
    report["quality_gate"] = quality_gate(report)
    report["result"] = "passed" if all(report["quality_gate"].values()) else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["result"] != "passed":
        raise SystemExit(f"MotionHub cache quality gate failed: {report['quality_gate']}")


if __name__ == "__main__":
    main()
