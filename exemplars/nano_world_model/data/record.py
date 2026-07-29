"""record.py — make your own data: play VizDoom and write pixel shards.

    [record.py]  ->  encode.py  ->  build_cache.py  ->  train_wm.py

The other path (data/download.py) borrows a public recording. This one needs
`pip install vizdoom`, and gives you two things that matter for a world model:

  * ACTIONS ARE GROUND TRUTH. You chose them, so there is no question of what id 7
    means. A downloaded set gives you someone else's labels and you get to hope.
  * DATA IS UNLIMITED. The model's ceiling stops being the size of a download.

WHAT THIS RECORDER PINS DOWN, and why each one is written rather than defaulted:

  1. THE SCENARIO is `scenarios/deathmatch_simple` (vendored — see NOTICE), the same
     map the downloaded PPO set was recorded on. VizDoom's own bundled `deathmatch`
     scenario is a DIFFERENT MAP: a model trained on one and evaluated on the other
     is being asked to predict a world it has never seen, and the loss says so.
  2. ONE RECORDED FRAME IS ONE GAME TIC (`frame_skip=1`). This has to match whatever
     produced any other data you mix in — cadence is not something the model can
     average out, it is the time axis it is learning.
  3. ACTION IDS ARE COMBOS OF SIX BUTTONS in product order, 18 of them, matching the
     downloaded set's ids exactly (ACTION_COMBOS below). Recording with a different
     ordering yields data that looks fine and trains a model conditioned on noise.

Storage is streaming: a raw uint8 memmap of frames plus a sidecar of per-frame
actions and episode ids. Clips are cut later, in encode.py. Storing clips here
instead would write each frame once per window it appears in — at stride 1 that is
17 copies of every frame.

    python -m exemplars.nano_world_model.data.record                  # 200k frames
    python -m exemplars.nano_world_model.data.record --frames 1_000_000 --seed 1

Budget frames, not minutes. Headless VizDoom runs about fifty times faster than the
game does, so a few minutes of wall clock is hours of footage and tens of GB on disk
— the number you actually have to plan around is frames. This prints the cost before
it starts.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from exemplars.nano_world_model import spec

SCENARIO = Path(__file__).resolve().parent / "scenarios" / "deathmatch_simple.cfg"

# The six buttons the scenario exposes, in the order its cfg declares them.
BUTTONS = ["ATTACK", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_RIGHT", "TURN_LEFT"]

# Action id -> button vector. The legal combinations of those six: ATTACK is
# exclusive, left/right strafe are mutually exclusive, left/right turn are mutually
# exclusive, and there is no no-op. That comes to exactly 18 — which is also how many
# ids the downloaded PPO set uses, and the two agree id for id.
#
# "Exactly 18" is a coincidence worth distrusting: an unrelated VizDoom gym wrapper
# also exposes 18 actions with completely different meanings, and its labels line up
# with these by arity alone. If you ever need to check which table a dataset used,
# do not compare counts — render a run of one id and look at it. Id 7 here is
# strafe-left + turn-right, which reads unmistakably as a wide circling motion.
ACTION_COMBOS = [
    [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0]]

# Actions worth spending most of the time on: turns, forward (weighted), forward-turns,
# forward-strafes, fire. A uniform draw would spend most frames in rare combinations
# and produce a dataset of twitching.
POLICY_POOL = [0, 1, 8, 8, 8, 9, 10, 11, 14, 17]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", default="rec")
    ap.add_argument("--frames", type=int, default=200_000,
                    help="frames to record (200k ~ 10GB at 128px, ~2 min)")
    ap.add_argument("--res", type=int, default=spec.RES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bots", type=int, default=8,
                    help="deathmatch bots per episode — an empty map teaches an empty world")
    ap.add_argument("--sticky", type=float, default=0.7,
                    help="probability of repeating the previous action, so motions sustain")
    ap.add_argument("--episode-tics", type=int, default=2100,
                    help="tics per episode before restarting (~60s at 35 tics/s)")
    args = ap.parse_args()

    import vizdoom as vzd
    from PIL import Image

    assert len(ACTION_COMBOS) == spec.N_ACTIONS, \
        f"{len(ACTION_COMBOS)} combos but spec.N_ACTIONS = {spec.N_ACTIONS}"

    game = vzd.DoomGame()
    game.load_config(str(SCENARIO))
    game.set_window_visible(False)          # headless; the cfg asks for a window
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(args.seed)
    game.init()

    def new_episode():
        game.new_episode()
        game.send_game_command("removebots")
        for _ in range(args.bots):
            game.send_game_command("addbot")

    spec.PIXEL_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = spec.PIXEL_SHARD_DIR / f"{args.tag}_{args.seed:04d}.bin"
    if bin_path.exists():
        raise SystemExit(f"{bin_path} exists — pick another --tag/--seed")

    gb = args.frames * args.res * args.res * 3 / 1e9
    print(f"recording {args.frames} frames at {args.res}px -> {bin_path.name}\n"
          f"  {gb:.1f} GB of pixels, ~{args.frames // args.episode_tics + 1} episodes\n"
          f"  (encode.py reads these and they can be deleted afterwards; codes are "
          f"~{gb / 60:.2f} GB)", flush=True)

    rng = np.random.default_rng(args.seed)
    actions, episodes = [], []
    ep, action, n, t0 = 0, POLICY_POOL[0], 0, time.time()

    # Frames are appended as they come; the file is exactly what was written.
    with open(bin_path, "wb") as f:
        new_episode()
        while n < args.frames:
            if game.is_episode_finished() or (n and n % args.episode_tics == 0):
                new_episode()
                ep += 1
            if rng.random() > args.sticky:
                action = int(rng.choice(POLICY_POOL))

            game.make_action(ACTION_COMBOS[action], 1)     # 1 tic — see docstring #2
            state = game.get_state()
            if state is None:                              # episode ended mid-action
                continue

            frame = state.screen_buffer                    # [H,W,3] uint8
            if frame.shape[0] != args.res or frame.shape[1] != args.res:
                frame = np.asarray(Image.fromarray(frame).resize(
                    (args.res, args.res), Image.BILINEAR), np.uint8)
            f.write(frame.tobytes())
            actions.append(action)
            episodes.append(ep)
            n += 1
            if n % 20000 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{args.frames} frames, {ep + 1} episodes, "
                      f"{rate:.0f} fps, {(args.frames - n) / rate:.0f}s left", flush=True)

    game.close()
    np.savez(str(bin_path).replace(".bin", "_meta.npz"),
             actions=np.asarray(actions, np.uint8),
             episode_id=np.asarray(episodes, np.int32),
             res=np.int32(args.res), seed=np.int32(args.seed),
             scenario=SCENARIO.name, frame_skip=np.int32(1))

    print(f"{n} frames / {ep + 1} episodes in {time.time() - t0:.0f}s -> {bin_path} "
          f"({os.path.getsize(bin_path) / 1e9:.2f} GB)")
    print("next: python -m exemplars.nano_world_model.data.encode --source recorded")


if __name__ == "__main__":
    main()
