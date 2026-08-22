"""
data/record/worlds.py — the world regime and the sidecar's entity ground truth.

This exemplar records ONE regime: 8 deathmatch bots, so the world moves on its
own. The other regime ids (sleeping/woken monster worlds, the kill world) are
research machinery that stayed with the research twin — the CONSTANTS remain
because the sidecar schema stores a world id per episode and the two recorders
share that schema.

`extract` is the ground-truth sidecar: visible labels + mover truth, per frame.
Nothing in training reads it; it is what lets you audit a corpus after the
fact, and it is cheap.
"""

MONSTERS = ("Zombieman", "ShotgunGuy", "DoomImp", "Demon", "Cacodemon", "HellKnight")
MOVER_CLASSES = set(MONSTERS) | {"DoomPlayer"}
TRANSIENT = {"TeleportFog", "Blood", "BulletPuff", "Rocket", "PlasmaBall"}

# The layer table: each layer picks a world regime; the ORDER is the encoding
# of ep_layer in the sidecar, shared with the research twin.
LAYERS = ["ent_revisit", "ent_long", "bots", "bots_long", "kill", "pans"]


# world regimes: KILLM = monster world with guns free (no notarget, monsters
# wake on damage) — the kill layer's controllable variant. The A1 sleeping-
# purity gate applies to WORLD_ASLEEP only.
WORLD_BOTS, WORLD_ASLEEP, WORLD_AWAKE, WORLD_KILLM = 0, 1, 2, 3

# arena interior. pipe3 used a walkabout-derived box (-330,690,-690,310) that
# BOTH undershot the real map (±700, see plot_travel) and included solid
# geometry — 13% of summons landed in never-walkable cells. ent2 fixes both:
# full bounds + an optional walkable-cell mask (recipe monsters.walkable_mask,
# produced by plot_travel --emit_mask from the previous corpus's poses).
ARENA = (-700, 700, -700, 700)          # x_lo, x_hi, y_lo, y_hi


def extract(state, self_id):
    """(labels, movers) for one state, class NAMES kept (ids are assigned by the
    writer). labels = visible non-transient entities EXCLUDING SELF (the
    player's own weapon sprite is labeled DoomPlayer — leaving it in made the
    event engine lock onto its own gun, smoke2 lesson); movers = monster/player
    ground truth incl. off-screen."""
    labs, movs = [], []
    for l in (state.labels or []):
        if l.object_name in TRANSIENT or l.object_id == self_id:
            continue
        labs.append((l.object_id, l.object_name,
                     l.x, l.y, l.width, l.height,
                     l.object_position_x, l.object_position_y))
    for o in (state.objects or []):
        if o.name in MOVER_CLASSES and o.id != self_id:
            movs.append((o.id, o.name, o.position_x, o.position_y))
    return labs, movs


# --- preroll drivers ---------------------------------------------------------

def _self_id(game):
    st = game.g.get_state()
    ids = [o.id for o in (st.objects or []) if o.name == "DoomPlayer"]
    assert len(ids) == 1, f"expected exactly one player pre-bots, got {ids}"
    return ids[0]
