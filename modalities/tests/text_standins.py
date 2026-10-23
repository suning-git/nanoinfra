"""Stand-ins for testing TextDataSource without a trained tokenizer or FineWeb: small parquet files whose documents
name themselves, a tokenizer that turns each into tokens that name their position, and the text vocab's layout and
control resolver. Imported by the text-source tests; not a test module itself.

A document's text is "f{file}g{row group}d{document}n{count}"; the stand-in tokenizer encodes it as BOS followed by
`count` tokens, token i being f * 10**9 + g * 10**6 + d * 10**3 + i + 1, so a window's text tokens say where each
came from (BOS is 0 everywhere: count positions, never token values).
"""

import re

import pyarrow as pa
import pyarrow.parquet as pq

from core.tokenization.vocab_layout import VocabLayout
from modalities.text.data_source import TextDataSource

BOS = 0
CONTROL = 10 ** 12                 # the control band starts above every stand-in text token
TEXT_TEMPLATE = {"template": ["bos", "text_start", "text_tokens", "text_end", "eos"], "supervise": "all"}


class Tokenizer:
    def get_bos_token_id(self):
        return BOS

    def get_vocab_size(self):
        return CONTROL + 4

    def get_special_tokens(self):
        return ["<|bos|>", "<|text_start|>", "<|text_end|>", "<|eos|>"]

    def encode(self, texts, prepend=None, num_threads=1):
        out = []
        for t in texts:
            f, g, d, n = map(int, re.fullmatch(r"f(\d+)g(\d+)d(\d+)n(\d+)", t).groups())
            out.append([prepend] + [f * 10 ** 9 + g * 10 ** 6 + d * 10 ** 3 + i + 1 for i in range(n)])
        return out


class Resolver:
    IDS = {"bos": CONTROL, "text_start": CONTROL + 1, "text_end": CONTROL + 2, "eos": CONTROL + 3}

    def resolve(self, name):
        return self.IDS.get(name)


def layout():
    lay = VocabLayout()
    lay.add_range(0, 0, CONTROL)
    lay.add_range(2, CONTROL, CONTROL + 4)
    return lay


def tokenizers():
    return {"text": Tokenizer(), "layout": layout(), "control_resolver": Resolver()}


def write_files(tmp_path, lengths):
    """lengths[f][g] = the token counts (without BOS) of row group g's documents in file f; returns the paths."""
    paths = []
    for f, groups in enumerate(lengths):
        path = tmp_path / f"shard_{f:05d}.parquet"
        writer = None
        for g, docs in enumerate(groups):
            table = pa.table({"text": [f"f{f}g{g}d{d}n{n}" for d, n in enumerate(docs)]})
            writer = writer or pq.ParquetWriter(path, table.schema)
            writer.write_table(table)              # one row group per call
        writer.close()
        paths.append(str(path))
    return paths


def doc_tokens(lengths, f, g, d):
    """The tokens of document d of row group g of file f, with BOS, as the stand-in tokenizer gives them."""
    return Tokenizer().encode([f"f{f}g{g}d{d}n{lengths[f][g][d]}"], prepend=BOS)[0]


def source(files, text_len, rank=None, world_size=None, **config):
    """A TextDataSource over `files`, windows of text_len text tokens (the recipe adds 4 control tokens)."""
    cfg = {"recipe": TEXT_TEMPLATE, "files": files, "sample_len": text_len + 4, "buffer_batch_size": 2,
           "tokenizer_batch_size": 2, **config}
    if rank is not None:
        cfg.update(rank=rank, world_size=world_size)
    return TextDataSource(cfg, tokenizers())


def text_of(sample, text_len):
    """The window's text tokens out of a sample [bos, text_start, text..., text_end, eos]."""
    return sample["tokens"][2:2 + text_len].tolist()
