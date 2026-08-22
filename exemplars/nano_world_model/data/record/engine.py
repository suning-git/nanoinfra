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
        g.init()
        g.new_episode()
        self.g = g
        self.tic = 0                      # completed make_action calls
        self.cmds = []                    # [(tic, cmd)] — sent before action `tic`
        self.pre_buttons = []             # preroll button vectors (uint8[6])
        self.in_preroll = True

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
            self.g.respawn_player()
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


