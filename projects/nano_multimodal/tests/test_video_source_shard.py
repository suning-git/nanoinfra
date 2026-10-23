"""VideoRowSource reads the share it is told; told nothing, the launcher's.

resolve_sources gives every source a rank and a world size: the process's own for training, rank 0 of 1 for the
validation set every rank freezes whole (train.py). Before 2026-10-21 the video source ignored them and its sampler
took the launcher's process, so on two GPUs each rank froze other validation rows. Pinned with a tiny cache in a
temporary directory (8 rows; every code of row i reads i), a launcher environment of rank 1 of 2 and no process group,
through the project's own assembly (a 9-frame 16 px clip). CPU only.

    pytest projects/nano_multimodal/tests/test_video_source_shard.py
"""

import json

import numpy as np
import pytest

from projects.nano_multimodal import assembly, spec
from projects.nano_multimodal.sources import VideoRowSource

ROWS, FRAMES, RES = 8, 9, 16


@pytest.fixture
def cache(tmp_path):
    shape = spec.video_shape(FRAMES, RES)
    meta = {"shape_contract": {**shape, "n_action_tokens": FRAMES}, "codec": spec.CODEC_NAME,
            "code_dtype": "uint16", "action_dtype": "uint8", "rows": {"train": ROWS, "val": ROWS}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    for split in ("train", "val"):
        np.repeat(np.arange(ROWS, dtype=np.uint16)[:, None], shape["code_len"], axis=1).tofile(
            tmp_path / f"{split}_codes.u16")
        np.zeros((ROWS, FRAMES), dtype=np.uint8).tofile(tmp_path / f"{split}_actions.u8")
    return tmp_path


@pytest.fixture
def rank_1_of_2(monkeypatch):
    for name, value in (("RANK", 1), ("LOCAL_RANK", 1), ("WORLD_SIZE", 2)):
        monkeypatch.setenv(name, str(value))


def _source(cache, share):
    """The val source as build_loader configures it; share=None drops rank and world_size from its config."""
    config = assembly.load_config("video")
    config["clip"] = {"frames": FRAMES, "res": RES}
    for key in ("sources", "val_sources"):
        config["data"][key][0]["cache"] = str(cache)
    vocab = assembly.assemble_vocab("video", config)
    sc = assembly.resolve_sources(config, vocab, "val", "cpu", *(share or (0, 1)))[0]
    sc = {k: v for k, v in sc.items() if k != "type" and not (share is None and k in ("rank", "world_size"))}
    return VideoRowSource(sc, vocab.tokenizers), vocab.layout.offset(spec.VIDEO_TYPE_ID)


def _rows_read(cache, share, n):
    src, video_offset = _source(cache, share)
    first_code = src._field[0] + int(src._code_slots[0])
    it = iter(src)
    return [int(next(it)["tokens"][first_code]) - video_offset for _ in range(n)]


def test_the_video_source_reads_the_share_it_is_told(cache, rank_1_of_2):
    assert sorted(_rows_read(cache, (0, 1), ROWS)) == list(range(ROWS))        # rank 0 of 1: every row, once
    mine, other = _rows_read(cache, (1, 2), ROWS // 2), _rows_read(cache, (0, 2), ROWS // 2)
    assert sorted(mine + other) == list(range(ROWS))                            # two halves of one epoch
    assert _rows_read(cache, None, ROWS // 2) == mine                           # told nothing: the launcher's
