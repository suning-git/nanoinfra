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

Every warp target goes through `sample_spot` (v4.1): inside the map with a
margin, on the walkable mask. `warp x y` itself checks nothing — outside the
map the frame stops being redrawn while the game keeps ticking — and the
runner drops any episode whose player is found outside (data/README.md,
"Recorder version").
"""

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.data.record.engine import NOOP

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

# Warp TARGETS keep this far from any wall (player radius 16; the picture
# already degrades 8 units past a wall). The ARENA box passes 82% -> 75%.
WALL_MARGIN = 32


def sample_spot(game, rng, mask, placed=(), min_dist=0, keep=None, tries=40):
    """One warp target: inside the map (`game.in_map`, WALL_MARGIN), on the
    walkable mask if any, >= min_dist from every (x, y) in `placed`, passing
    `keep(x, y)` if given. None if all `tries` fail. The ARENA box is a box;
    the map is an octagon with notches (18% of the box is void), and the
    walkable mask cannot stand in for this test — it is built from a previous
    corpus's own poses, which had the same bug (v4.1, 2026-08-31)."""
    x_lo, x_hi, y_lo, y_hi = ARENA
    for _ in range(tries):
        x, y = int(rng.integers(x_lo, x_hi)), int(rng.integers(y_lo, y_hi))
        if not game.in_map(x, y, WALL_MARGIN):
            continue
        if mask is not None and not mask.ok(x, y):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_dist ** 2 for px, py in placed):
            continue
        if keep is not None and not keep(x, y):
            continue
        return x, y
    return None


def bot_positions(game, self_id):
    """(x, y) of every other player (the bots) right now — the `placed` list a
    bots-world home warp must keep away from."""
    st = game.g.get_state()
    return [(float(o.position_x), float(o.position_y))
            for o in (st.objects or []) if o.name == "DoomPlayer" and o.id != self_id]


def warp_home(game, rng, mask, placed, min_dist=160):
    """Warp the player to a home spot before the stream starts: away from
    everything in `placed`, INSIDE THE MAP, and mobile (a blind warp can wedge
    the player into solid geometry — smoke2 spent a whole episode staring at a
    wall). Probe: 6 FWD tics must move >= 10 units. The mobility probe alone is
    ANTI-diagnostic for the void — nothing out there blocks you, so it always
    passed — which is how 166 pipe4 episodes started outside the map. Hence
    in_map on the landed pose, before and after the probe. No valid home in 6
    rounds = no episode (it used to proceed with whatever the last warp gave).
    Then 30 NOOP tics so whatever was placed settles before frame 0.

    v4.2 (2026-09-14): monster worlds always did this inside seed_monsters;
    BOTS WORLDS DID NOT — the player stayed on the wad's type-1 start while the
    8 bots spawned on top of him, and the first ~40 recorded frames (median;
    up to 373) were bot bodies / spawn sprites at point-blank range. 858/1732
    pipe4 episodes (every bots-world episode) begin that way: 63,479 frames,
    0.78% of the corpus, invisible to the out-of-map health check. Returns
    True on success, False = do not record this episode."""
    fwd = spec.ACTION_COMBOS[8]
    for _ in range(6):
        spot = sample_spot(game, rng, mask, placed, min_dist=min_dist)
        if spot is None:
            continue
        x, y = spot
        game.cmd(f"warp {x} {y}")
        if game.step_vec(NOOP) is None:
            return False
        hx, hy, _ = game.pose()
        if not game.in_map(hx, hy):
            continue
        for _ in range(6):
            if game.step_vec(fwd) is None:
                return False
        mx, my, _ = game.pose()
        if (mx - hx) ** 2 + (my - hy) ** 2 >= 10 ** 2 and game.in_map(mx, my):
            break
    else:
        return False
    for _ in range(30):
        if game.step_vec(NOOP) is None:
            return False
    return True


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
