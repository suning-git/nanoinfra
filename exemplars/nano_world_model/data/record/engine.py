"""
data/record/engine.py — the VizDoom shell and where bytes land.

Everything in here is REPLAY-CRITICAL: `Game` records the console-command
schedule and the preroll button vectors alongside the stream, which is what
makes an episode reconstructible from its sidecar alone (gate D1, bit-exact).
Change the order of commands or steps and old corpora stop replaying.

Paths are resolved ONCE here so every consumer agrees on them: this exemplar
has exactly one corpus home (spec.DATASET_ROOT), so recorder and consumers
land in the same place by construction.
"""

import io
import os

import numpy as np

from exemplars.nano_world_model import spec

def data_root(recipe):
    return spec.REPO / "datasets" / recipe.get("root", "nano_world_model")


# CONSUMER-side constants (one spelling, owned by spec).
DATA_ROOT = spec.DATASET_ROOT
BUFFER = spec.PIXEL_SHARD_DIR
SIDECARS = spec.SIDECAR_DIR

SCEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scenarios", "deathmatch_simple.cfg")

H, W = 240, 320
NOOP = np.zeros(6, np.uint8)

# Stamped into every sidecar (`recorder_version`) so a corpus says which
# recorder made it. v4.1 (2026-08-31): every warp target and every recorded
# tic must be inside the map (see MapBounds / Game.in_map); auto-respawns are
# recorded as `ep_respawn_tics`. Corpora from v4 and earlier can contain
# episodes recorded entirely outside the map — filter them with MapBounds.
# v4.2 (2026-09-14): bots worlds get the home warp too — earlier corpora begin
# every bots-world episode with ~40 frames of bots spawning on the player.
RECORDER_VERSION = "v4.2"

class MapBounds:
    """The playable area as the engine reports it: the ONE-SIDED wall lines,
    which are exactly the boundary between map and void (solid pillars are
    boundary too; two-sided lines — room partitions, and railings that block
    movement — are not). Built from `state.sectors` (needs
    `set_sectors_info_enabled`), so it follows whatever scenario wad is
    loaded with no parser of our own."""

    def __init__(self, walls):
        self.walls = np.asarray(walls, np.float64).reshape(-1, 4)  # x1,y1,x2,y2
        assert len(self.walls), "no wall lines — sectors info off?"

    @classmethod
    def from_state(cls, state):
        # ViZDoom lists a line under every sector it borders: a two-sided line
        # appears twice, a one-sided (boundary) line exactly once. Selecting
        # on multiplicity rather than `is_blocking` keeps a blocking two-sided
        # line (a railing) from flipping the parity of everything behind it.
        # Known residual: a self-referencing line (both sides face the same
        # sector — ZDoom's deep-water idiom) is listed once and would count
        # as boundary; this wad has none (56 lines: 32 once, 24 twice).
        count = {}
        for sec in (state.sectors or []):
            for l in sec.lines:
                k = (float(l.x1), float(l.y1), float(l.x2), float(l.y2))
                count[k] = count.get(k, 0) + 1
        return cls(sorted(k for k, n in count.items() if n == 1))

    def inside(self, x, y, margin=0.0):
        """Even-odd ray cast over the wall segments, then (if margin > 0) the
        distance from (x, y) to the nearest segment must be >= margin.
        A point exactly ON a wall line is a tie the ray cast breaks either
        way; the player's 16-unit radius keeps recorded poses off wall lines,
        and warp targets carry a margin, so no caller sits on the tie."""
        x1, y1, x2, y2 = self.walls.T
        dy = np.where(y2 == y1, 1.0, y2 - y1)      # horizontal walls never cross
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / dy + x1)
        if int(cross.sum()) % 2 == 0:
            return False
        if margin > 0:
            bx, by = x2 - x1, y2 - y1
            t = np.clip(((x - x1) * bx + (y - y1) * by) / (bx * bx + by * by), 0, 1)
            if float(np.hypot(x1 + t * bx - x, y1 + t * by - y).min()) < margin:
                return False
        return True


class WalkableMask:
    def __init__(self, path):
        d = np.load(path)
        self.grid, self.origin, self.cell = d["grid"], float(d["origin"]), float(d["cell"])

    def ok(self, x, y):
        cx = int((x - self.origin) // self.cell)
        cy = int((y - self.origin) // self.cell)
        return (0 <= cx < self.grid.shape[0] and 0 <= cy < self.grid.shape[1]
                and bool(self.grid[cx, cy]))


class Game:
    """One ViZDoom instance = ONE EPISODE (live reseed is not reproducible —
    2026-07-24 probe). Records the console-command schedule and preroll button
    vectors so the whole episode replays bit-exactly from the sidecar."""

    def __init__(self, seed, timeout):
        import vizdoom as vzd
        self.vzd = vzd
        g = vzd.DoomGame()
        g.load_config(SCEN)
        g.set_window_visible(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_seed(seed)
        g.set_episode_timeout(timeout)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        # Sectors info feeds MapBounds only. It is state extraction, not game
        # logic, so replaying a pre-v4.1 sidecar stays bit-exact (test 5 of
        # tests/test_recorder_v41.py).
        g.set_sectors_info_enabled(True)
        g.init()
        g.new_episode()
        self.g = g
        self.tic = 0                      # completed make_action calls
        self.cmds = []                    # [(tic, cmd)] — sent before action `tic`
        self.pre_buttons = []             # preroll button vectors (uint8[6])
        self.in_preroll = True
        self.respawn_tics = []            # tics at which step_vec auto-respawned
        self.bounds = MapBounds.from_state(g.get_state())

    def in_map(self, x, y, margin=0.0):
        """Is (x, y) inside the playable map, at least `margin` units from any
        wall? Every warp target and every recorded tic must pass this: `warp`
        succeeds at any coordinate, and outside the map one-sided walls are
        invisible from behind, so the frame stops being redrawn while the
        engine keeps ticking (9.6% of pipe4's episodes started that way)."""
        return self.bounds.inside(x, y, margin)

    def cmd(self, c):
        self.cmds.append((self.tic, c))
        self.g.send_game_command(c)

    def step_vec(self, vec):
        """One tic with a raw button vector. Returns the post-step state (or
        None if the engine ended the episode)."""
        if self.in_preroll:
            self.pre_buttons.append(np.asarray(vec, np.uint8))
        self.g.make_action([float(v) for v in vec], 1)
        self.tic += 1
        if self.g.is_player_dead():
            self.g.respawn_player()       # lands on the player start; recorded
            self.respawn_tics.append(self.tic)   # (the flag reads False again)
        if self.g.is_episode_finished():
            return None
        return self.g.get_state()

    def step(self, action_id):
        return self.step_vec(spec.ACTION_COMBOS[action_id])

    def pose(self):
        v = self.vzd.GameVariable
        return (self.g.get_game_variable(v.POSITION_X),
                self.g.get_game_variable(v.POSITION_Y),
                self.g.get_game_variable(v.ANGLE))

    def close(self):
        self.g.close()


def jpeg_frame(state):
    """The v2 byte path: native 240x320 -> JPEG q85 round-trip."""
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(state.screen_buffer).save(b, "JPEG", quality=85)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))


