"""Train the text tokenizer artifact — the modality's artifact-build orchestrator.

Produces the two files everything else loads (see get_tokenizer / get_token_bytes):

  <out>/tokenizer.pkl     rustbpe-trained BPE (tiktoken encoding), control tokens
                          appended as the LAST len(CONTROL_TOKENS) ids
  <out>/token_bytes.pt    int64 [vocab] — UTF-8 byte length per token id, control
                          band = 0 (the bits-per-byte denominator)

Corpus = the FineWeb parquet shards already on disk under base_data/ (the same
files training streams; download via exemplars/text_pretrain/data/download_shards.py).
This is the assembly point of the artifact-build flow (D1.3): the protocol
registry is imported from modalities/control and passed as display-form strings
into the protocol-agnostic trainer — the tokenizer itself knows nothing of
"control", it just appends the strings it is handed.

  # the standard artifact (matches the exemplar world): vocab 32768, seconds
  .venv/bin/python -m modalities.text.train_tokenizer

  # a quick small-vocab world for experiments
  .venv/bin/python -m modalities.text.train_tokenizer --vocab-size 4096 --train-chars 5000000 --out /tmp/tok

Refuses to overwrite an existing artifact unless --force: every trained checkpoint
depends on the EXACT vocab, so silently replacing tokenizer.pkl orphans them.

(Crystallized from the rebuild-era prepare_data.py's tokenizer half — the birth
entry of the original artifact; see exemplars/text_pretrain/provenance.md.)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

from core.utils import get_base_dir
from modalities.control import CONTROL_TOKENS, display_form
from modalities.text.fineweb import list_parquet_files
from modalities.text.tokenizer import RustBPETokenizer


def iter_parquet_texts(max_chars: int, data_dir: str | None = None):
    """Yield document strings from the local parquet shards until max_chars consumed."""
    total = 0
    for path in list_parquet_files(data_dir):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for txt in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                if not txt:
                    continue
                yield txt
                total += len(txt)
                if total >= max_chars:
                    return


def build_token_bytes(tok: RustBPETokenizer) -> torch.Tensor:
    """int64 [vocab]: UTF-8 byte length per id; control band = 0 (bpb denominator)."""
    offset = tok.control_token_offset
    n = tok.get_vocab_size()
    tb = torch.zeros(n, dtype=torch.int64)
    for i in range(offset):
        tb[i] = len(tok.enc.decode_single_token_bytes(i))
    return tb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocab-size", type=int, default=32768,
                    help="TOTAL vocab incl. the control band (default: the exemplar world's 32768)")
    ap.add_argument("--train-chars", type=int, default=40_000_000,
                    help="chars of corpus fed to BPE training")
    ap.add_argument("--data-dir", default=None,
                    help="parquet shard dir (default: $NANOINFRA_DATA_DIR or base_data/)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: $NANOINFRA_TOKENIZER_DIR or <base_dir>/tokenizer)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing tokenizer.pkl (orphans checkpoints trained on it)")
    args = ap.parse_args()

    out_dir = Path(args.out or os.environ.get("NANOINFRA_TOKENIZER_DIR")
                   or os.path.join(get_base_dir(), "tokenizer"))
    pkl = out_dir / "tokenizer.pkl"
    if pkl.exists() and not args.force:
        sys.exit(f"REFUSING: {pkl} already exists — trained checkpoints depend on the exact "
                 f"vocab. Pass --force to replace it, or --out for a separate directory.")

    specials = [display_form(n) for n in CONTROL_TOKENS]
    print(f"training rustbpe tokenizer: vocab={args.vocab_size} "
          f"({args.vocab_size - len(specials)} BPE + {len(specials)} control), "
          f"<= {args.train_chars/1e6:.0f}M chars from parquet ...")
    tok = RustBPETokenizer.train_from_iterator(
        iter_parquet_texts(args.train_chars, args.data_dir), args.vocab_size, specials)
    assert tok.get_vocab_size() == args.vocab_size, \
        f"trained vocab {tok.get_vocab_size()} != requested {args.vocab_size}"

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir))
    torch.save(build_token_bytes(tok), out_dir / "token_bytes.pt")
    print(f"token_bytes.pt written ({args.vocab_size} entries, control band = 0)")
    print(f"DONE -> {out_dir}  (loaded automatically by get_tokenizer; "
          f"set NANOINFRA_TOKENIZER_DIR to use a non-default location)")
    # rustbpe's C extension can abort during interpreter finalization even after
    # full success (PyGILState noise) -> flush and hard-exit 0, same as the
    # original prepare_data.py did.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
