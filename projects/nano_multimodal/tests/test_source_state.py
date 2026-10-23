"""VideoRowSource's state reads back from JSON unchanged (see core.data.DataSource.get_state). The source is given
only what get_state reads (its sampler), so no clip cache is needed.

    pytest projects/nano_multimodal/tests/test_source_state.py
"""
import json

from core.data.dist_sampler import ResumableDistributedSampler
from projects.nano_multimodal.sources import VideoRowSource


def test_the_video_row_source_state_reads_back():
    src = object.__new__(VideoRowSource)
    src._sampler = ResumableDistributedSampler(range(8), seed=0)
    it = iter(src._sampler)
    for _ in range(5):
        next(it)
    state = src.get_state()
    assert json.loads(json.dumps(state)) == state
