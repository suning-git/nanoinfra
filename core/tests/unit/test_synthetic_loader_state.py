"""The synthetic loader's state reads back from JSON unchanged (see DataSource.get_state).

It is the text source's state when the text config gives a `data_path` (modalities/text/fineweb.token_data_loader).
"""

import json

from core.data.synthetic_loader import synthetic_token_loader


def _reads_back(state):
    assert json.loads(json.dumps(state)) == state


def test_the_synthetic_loader_state_reads_back(tmp_path):
    _reads_back(next(synthetic_token_loader(2, 8, device="cpu", vocab_size=64))["state_dict"])
    stream = tmp_path / "tokens.bin"
    stream.write_bytes(bytes(range(40)))
    it = synthetic_token_loader(2, 8, device="cpu", data_path=str(stream), vocab_size=256)
    for _ in range(3):                                     # past the end of the stream once
        batch = next(it)
    _reads_back(batch["state_dict"])
