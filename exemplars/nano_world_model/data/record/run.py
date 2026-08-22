"""
data/record/run.py — the episode runner and the CLI.

One episode = one fresh DoomGame (live reseeding is not reproducible, probed
2026-07-24). The runner owns the things that must hold in EVERY mode: the
stuck-escape ladder (turn -> forward, then a replay-safe warp), the wall probe
that keeps dwells and forward runs off wall faces, and the per-tic write.

    python -m exemplars.nano_world_model.data.record.run \
        --tag p4a --minutes 30 --seed 1 --recipe .../recipes/pipe4.yaml
"""

import argparse
import os
import time

import numpy as np
import yaml

from exemplars.nano_world_model import spec
from exemplars.nano_world_model.data.record import policies
from exemplars.nano_world_model.data.record.engine import (
    NOOP, Game, WalkableMask, data_root, jpeg_frame)
from exemplars.nano_world_model.data.record.policies import (
    FWD_FAMILY, BalancedChooser, PansPolicy, coverage_policy)
from exemplars.nano_world_model.data.record.shards import ShardWriter
from exemplars.nano_world_model.data.record.worlds import (
    ARENA, LAYERS, WORLD_ASLEEP, WORLD_BOTS, _self_id, extract)

# --- episode runner ----------------------------------------------------------

def run_episode(writer, rng, recipe, ep, seed, layer_name, world, n_frames,
                is_val, seg_chooser):
    layer = LAYERS.index(layer_name)
    timeout = n_frames + 1200             # covers preroll + stream, any layer
    game = Game(seed, timeout)
    game.cmd("removebots")
    if game.step_vec(NOOP) is None:
        game.close()
        return 0
    self_id = _self_id(game)

    # Bot worlds only. The research twin's other regimes (sleeping/woken
    # monster worlds, the kill world) are missions this exemplar does not
    # carry; the constants remain in worlds.py because the sidecar schema
    # stores a world id.
    if world != WORLD_BOTS:
        raise ValueError("this recorder ships bot worlds only — set "
                         "pans.world_bots_frac: 1.0 in the recipe")
    spawns = []
    for _ in range(8):
        game.cmd("addbot")
    if game.step_vec(NOOP) is None:
        game.close()
        return 0

    game.in_preroll = False
    filler = coverage_policy(rng, recipe, seg_chooser)
    targets = {"DoomPlayer"}
    # One actor: the segment library, uninterrupted. The research twin
    # dispatches per layer (events, kill); here every layer IS the library.
    actor = PansPolicy(filler, rng)

    # dwell wall-avoidance (ep104 fix): ray-march the walkable mask along the
    # facing; a blocked cell within ~120 units means a dwell would stare into
    # a wall. Mask = union of previously-walked cells, so unseen-but-open
    # reads "wall" — conservative in the right direction for dwells.
    wmask = (WalkableMask(recipe["monsters"]["walkable_mask"])
             if recipe.get("monsters", {}).get("walkable_mask") else None)

    def wall_probe():
        """(near, far): solid geometry within 48 / within 120 units of the
        facing ray. far gates dwells; near gates forward motion (a FWD ram is
        the same full-screen wall stall, small-batch audit #2)."""
        if wmask is None:
            return False, False
        px, py, ang = game.pose()
        r = np.radians(ang)
        blocked = [not wmask.ok(px + d * np.cos(r), py + d * np.sin(r))
                   for d in (48, 84, 120)]
        return blocked[0], any(blocked)

    st = game.g.get_state()
    labs, movs = extract(st, self_id)     # describes the CURRENT visible state
    n = 0
    recovery, pose_hist, stuck_rounds = [], [], 0
    blocked_run, last_xy = 0, game.pose()[:2]
    while n < n_frames and st is not None:
        if recovery:
            a = recovery.pop(0)
        else:
            near, far = wall_probe()
            # Displacement-truth blocked detector (audit #4: 77% of wall
            # stalls were FWD rams the mask cannot see — 45-unit cells
            # straddling a wall face read walkable). Held forward + no motion
            # for 3 tics = blocked, no mask involved.
            a = actor.act([(i, c, x, y, w, h) for (i, c, x, y, w, h, _, _) in labs
                           if c in targets], n, None,
                          wall_close=far, wall_near=near or blocked_run >= 3)
        nxt = game.step(a)
        if nxt is None:
            actor.abort()
            break
        labs, movs = extract(nxt, self_id)
        writer.add(jpeg_frame(nxt), a, game.pose(), labs, movs, ep, layer)
        xy = game.pose()[:2]
        if a in FWD_FAMILY and np.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) < 2.0:
            blocked_run += 1
        else:
            blocked_run = 0
        last_xy = xy
        st = nxt
        n += 1
        if n % 25 == 0:                   # wedged-in-geometry escape (smoke2)
            px, py, _ = game.pose()
            pose_hist.append((px, py))
            # v4: the escape runs in EVERY mode (v3 guarded it to filler and
            # 8/598 episodes spent 500+ tics pinned inside events). Window is
            # 200 tics, not 25: noop dwells are LEGITIMATE stillness now, and
            # only multi-dwell-length immobility marks a genuine wedge.
            if len(pose_hist) >= 8 and not recovery:
                ox, oy = pose_hist[-8]
                if (px - ox) ** 2 + (py - oy) ** 2 < 10 ** 2:
                    if actor.mode != "filler":
                        actor.abort()      # drop the in-flight event, then escape
                    stuck_rounds += 1
                    if stuck_rounds >= 3:
                        # A TRUE geometry wedge does not yield to turn+FWD: the
                        # pipe4 mass batch had one episode spend 8834 tics inside
                        # a 10x10 box while the escape swept a full 360 deg and
                        # held forward the whole time (33% FWD / 57% turn, 27
                        # units walked). Teleport is the only exit — and it is
                        # REPLAY-SAFE: cmds carry their tic and rerender re-issues
                        # them mid-stream (probed bit-exact, 2026-08-03). The jump
                        # is a hard visual cut, so it is deliberately the LAST
                        # resort: ~600 tics of failed escapes first, and the warp
                        # tic is in the cmd schedule for any consumer that wants
                        # to drop windows straddling it.
                        for _ in range(40):
                            wx = int(rng.integers(ARENA[0], ARENA[1]))
                            wy = int(rng.integers(ARENA[2], ARENA[3]))
                            if wmask is None or wmask.ok(wx, wy):
                                break
                        game.cmd(f"warp {wx} {wy}")
                        recovery = [policies.FWD] * 10
                        stuck_rounds = 0
                    else:
                        turn = policies.TR if rng.random() < 0.5 else policies.TL
                        recovery = [turn] * 40 + [policies.FWD] * 25
                else:
                    stuck_rounds = 0
    writer.note_episode(ep, seed=seed, val=is_val, world=world, layer=layer,
                        self_id=self_id, timeout=timeout, cmds=game.cmds,
                        pre_buttons=game.pre_buttons, spawns=spawns,
                        events=actor.events)
    game.close()
    return n


def dir_gb(p):
    """Un-encoded pixel backlog on disk, in GB. Uses ALLOCATED blocks, not
    st_size: an open shard is a sparse memmap that reports its full 11.5GB cap
    from the first frame, so three live recorders showed ~35GB of phantom
    backlog. On 2026-08-03 that phantom (plus three orphan .bin files left by
    a crash) tipped a 100GB cap and stalled every recorder while the audit gate
    was still waiting for their second shard — a deadlock made of accounting."""
    return (sum(f.stat().st_blocks * 512 for f in p.glob("*.bin")) / 1e9
            if p.exists() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--recipe", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "recipes", "minrec.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--buffer_cap_gb", type=float, default=150.0,
                    help="sleep while un-encoded pixel shards exceed this")
    args = ap.parse_args()

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)
    rng = np.random.default_rng(args.seed)
    writer = ShardWriter(args.tag, recipe)
    layer_chooser = BalancedChooser(recipe["layers"], rng)
    seg_chooser = BalancedChooser(recipe["coverage"]["segments"], rng)

    t0, ep, total, n_events = time.time(), 0, 0, 0
    while time.time() - t0 < args.minutes * 60 and \
            (not args.max_frames or total < args.max_frames):
        while dir_gb(writer.buffer) > args.buffer_cap_gb:
            print(f"[{args.tag}] buffer over {args.buffer_cap_gb}GB — waiting "
                  f"for the encoder", flush=True)
            time.sleep(30)
        layer = layer_chooser.pick()
        seed = int(rng.integers(1, 2 ** 31 - 1))
        every = recipe["val"]["episode_every"]
        is_val = (ep % every) == every - 1
        n_frames = recipe["episode"][f"{layer}_frames"]
        if layer == "pans":
            world = WORLD_BOTS if rng.random() < recipe["pans"]["world_bots_frac"] \
                else WORLD_ASLEEP
        else:
            world = WORLD_BOTS
        n = run_episode(writer, rng, recipe, ep, seed, layer, world, n_frames,
                        is_val, seg_chooser)
        layer_chooser.credit(layer, max(n, 1))
        n_events += len(writer.ep_meta.get(ep, {}).get("events", []))
        writer.end_episode()
        total += n
        ep += 1
        if ep % 5 == 0:
            el = time.time() - t0
            share = {k: f"{v:.2f}" for k, v in
                     zip(layer_chooser.names,
                         layer_chooser.done / max(layer_chooser.done.sum(), 1))}
            print(f"[{args.tag}] ep {ep} total {total} f  {total / el:.0f} f/s  "
                  f"events {n_events}  layers {share}", flush=True)
    writer.close()
    print(f"[{args.tag}] DONE: {total} frames, {ep} episodes, {n_events} events, "
          f"{total / (time.time() - t0):.0f} f/s", flush=True)


if __name__ == "__main__":
    main()
