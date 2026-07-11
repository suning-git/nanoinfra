"""Download additional FineWeb sample-10BT parquet shards into outputs/base_data/.

Reuses the original download mechanism (huggingface_hub.hf_hub_download into the
_hf/ cache dir, then move + rename with the `shard_NNN_00000.parquet` prefix so
files sort AFTER the existing shards). All-but-last sorted shard = train, last =
val (modalities/text/fineweb.py convention), so the new last shard becomes val.

Run: .venv/bin/python exemplars/text_pretrain/download_shards.py 003 004 005
"""
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceFW/fineweb"
BASE = Path("outputs/base_data")
CACHE = BASE / "_hf"


def main(idxs):
    for idx in idxs:
        hf_name = f"{idx}_00000.parquet"
        dst = BASE / f"shard_{idx}_00000.parquet"
        if dst.exists():
            print(f"[skip] {dst} already exists")
            continue
        print(f"[download] sample/10BT/{hf_name} ...", flush=True)
        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"sample/10BT/{hf_name}",
            local_dir=str(CACHE),
        )
        # verify integrity before publishing to base_data
        f = pq.ParquetFile(local)
        nrows = f.metadata.num_rows
        assert nrows > 0, f"empty parquet {local}"
        print(f"[verify] {hf_name}: rows={nrows:,} row_groups={f.num_row_groups} OK", flush=True)
        shutil.move(local, dst)
        print(f"[done] -> {dst}", flush=True)
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["003", "004", "005"])
