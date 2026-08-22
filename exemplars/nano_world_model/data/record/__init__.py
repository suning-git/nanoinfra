"""
data/record/ — making a corpus: the engine, the actors, the writer.

    engine.py    VizDoom shell, frame bytes, paths, walkable mask
    shards.py    rotating pixel shards + the durable sidecar
    policies.py  the actors: segment library with coverage quotas
    run.py       episode runner (stuck escape, wall probe) + CLI
    worlds.py    the bot world + entity ground truth for the sidecar

Ported from the research twin this exemplar's training stack comes from, and
stripped to the subset a corpus recipe like data/recipes/minrec.yaml
activates: bot worlds and the segment library. The research-only machinery
(monster worlds, the revisit event engine, the kill policy) stayed behind;
the sidecar schema is UNCHANGED, so tools that read sidecars work on both.
The port was verified bit-exact: same recipe, same seed, both recorders —
identical pixel shards and sidecars, before and after the strip.
"""

from exemplars.nano_world_model.data.record.engine import (  # noqa: F401
    BUFFER, DATA_ROOT, SIDECARS, H, W, NOOP, SCEN, Game, WalkableMask,
    data_root, jpeg_frame)
from exemplars.nano_world_model.data.record.shards import (  # noqa: F401
    ShardWriter)
from exemplars.nano_world_model.data.record.worlds import (  # noqa: F401
    ARENA, LAYERS, MONSTERS, MOVER_CLASSES, TRANSIENT, WORLD_BOTS, extract)
# run.py is not imported here: it is the CLI entry point, and importing it
# again under `python -m ...record.run` would warn about double import.
