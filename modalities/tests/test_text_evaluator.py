"""TextEvaluator's metric names: the stream's CE and BPB names come from its config, and as keys of one metrics dict
the second would silently replace the first — the stream would report its BPB under the CE's name and no CE. The
evaluator refuses equal names when it is built (before the byte table is fetched). Also: each process evaluates
its share of eval_tokens, the world size coming from the launcher's environment."""

import pytest

from modalities.text.evaluator import TextEvaluator


def test_equal_ce_and_bpb_names_are_refused():
    stream = {"name": "fineweb", "files": [], "recipe": None, "metric": "val/x", "bpb_metric": "val/x"}
    with pytest.raises(ValueError, match="metric and bpb_metric are both"):
        TextEvaluator(stream, {}, device_batch_size=2, sequence_len=8)


def test_each_process_reads_its_share_of_the_eval_tokens(monkeypatch):
    # world size from the launcher's environment (core.utils.process_identity), known before any process group
    stream = {"name": "fineweb", "files": [], "recipe": None}
    steps = []
    for world in (1, 2):
        for name, value in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", world)):
            monkeypatch.setenv(name, str(value))
        steps.append(TextEvaluator(stream, {"eval_tokens": 64}, device_batch_size=2, sequence_len=8)._eval.eval_steps)
    assert steps == [4, 2]
