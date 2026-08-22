"""
data/record/shards.py — rotating pixel shards + the durable sidecar.

Pixels are TRANSIENT (the encoder deletes them once codes exist); the sidecar
npz is PERMANENT — it carries the action stream, pose, entity truth, the
episode table and the replay schedule, which is what lets any window be
re-derived later without re-recording.
"""

import json

import numpy as np

from exemplars.nano_world_model.data.record.engine import H, W, data_root

class ShardWriter:
    """Rotating pixel shards + sidecar npz. Pixels are transient (encoder
    deletes after consumption); sidecars are durable."""

    def __init__(self, tag, recipe, cap=50_000):
        self.buffer = data_root(recipe) / "record_buffer"
        self.sidecars = data_root(recipe) / "sidecars"
        self.buffer.mkdir(parents=True, exist_ok=True)
        self.sidecars.mkdir(parents=True, exist_ok=True)
        self.tag, self.cap = tag, cap
        self.recipe = recipe
        self.cls = {}                     # name -> id (per writer, stored per shard)
        self.shard_i = -1
        self._roll()

    def cls_id(self, name):
        if name not in self.cls:
            self.cls[name] = len(self.cls)
        return self.cls[name]

    def _roll(self):
        if self.shard_i >= 0:
            self.close()
        self.shard_i += 1
        self.name = f"{self.tag}_{self.shard_i:04d}"
        self.buf = np.memmap(self.buffer / (self.name + ".bin"), dtype=np.uint8,
                             mode="w+", shape=(self.cap, H, W, 3))
        self.n = 0
        self.acts, self.pose, self.eps, self.layer = [], [], [], []
        self.lab_ofs, self.lab = [0], []
        self.mov_ofs, self.mov = [0], []
        self.ep_meta = {}                 # ep -> dict

    def add(self, frame, action, pose, labs, movs, ep, layer):
        assert self.n < self.cap, "shard overflow — end_episode() not called?"
        self.buf[self.n] = frame
        self.n += 1
        self.acts.append(action)
        self.pose.append(pose)
        self.eps.append(ep)
        self.layer.append(layer)
        self.lab += [(i, self.cls_id(c), x, y, w, h, wx, wy)
                     for (i, c, x, y, w, h, wx, wy) in labs]
        self.lab_ofs.append(len(self.lab))
        self.mov += [(i, self.cls_id(c), x, y) for (i, c, x, y) in movs]
        self.mov_ofs.append(len(self.mov))

    def note_episode(self, ep, **meta):
        self.ep_meta[ep] = meta

    def end_episode(self):
        if self.n >= self.cap - 25_000:
            self._roll()

    def close(self):
        if self.n == 0:                   # empty pre-allocated roll — remove it
            del self.buf
            (self.buffer / (self.name + ".bin")).unlink(missing_ok=True)
            return
        self.buf.flush()
        del self.buf
        with open(self.buffer / (self.name + ".bin"), "r+b") as f:
            f.truncate(self.n * H * W * 3)
        eps_here = sorted(set(self.eps))
        M = [self.ep_meta[e] for e in eps_here]
        lab = np.array(self.lab, np.float64).reshape(-1, 8)
        mov = np.array(self.mov, np.float64).reshape(-1, 4)
        events = [ev for m in M for ev in m["events"]]
        np.savez_compressed(
            self.sidecars / (self.name + "_sc.npz"),
            # per-frame stream
            actions=np.array(self.acts, np.int16),
            pose=np.array(self.pose, np.float32),
            episode_id=np.array(self.eps, np.int32),
            layer=np.array(self.layer, np.int8),
            # ragged: visible labeled entities
            lab_ofs=np.array(self.lab_ofs, np.int64),
            lab_id=lab[:, 0].astype(np.int32),
            lab_cls=lab[:, 1].astype(np.int16),
            lab_bbox=lab[:, 2:6].astype(np.uint16),
            lab_wpos=lab[:, 6:8].astype(np.float32),
            # ragged: mover ground truth (on- or off-screen)
            mov_ofs=np.array(self.mov_ofs, np.int64),
            mov_id=mov[:, 0].astype(np.int32),
            mov_cls=mov[:, 1].astype(np.int16),
            mov_pos=mov[:, 2:4].astype(np.float32),
            # episode table
            episodes=np.array(eps_here, np.int32),
            ep_seed=np.array([m["seed"] for m in M], np.int64),
            ep_val=np.array([m["val"] for m in M], bool),
            ep_world=np.array([m["world"] for m in M], np.int8),
            ep_layer=np.array([m["layer"] for m in M], np.int8),
            ep_self_id=np.array([m["self_id"] for m in M], np.int32),
            ep_timeout=np.array([m["timeout"] for m in M], np.int32),
            ep_pre_len=np.array([len(m["pre_buttons"]) for m in M], np.int32),
            pre_buttons=(np.concatenate(
                [np.asarray(m["pre_buttons"], np.uint8).reshape(-1, 6)
                 for m in M]) if any(len(m["pre_buttons"]) for m in M)
                else np.zeros((0, 6), np.uint8)),
            ep_cmds=json.dumps({int(e): m["cmds"] for e, m in zip(eps_here, M)}),
            spawns=json.dumps({int(e): m["spawns"] for e, m in zip(eps_here, M)}),
            # events (stream-frame indices, relative to episode start)
            ev_ep=np.array([e for m, e0 in zip(M, eps_here) for e in [e0] * len(m["events"])],
                           np.int32),
            ev_ent=np.array([ev.ent_id for ev in events], np.int32),
            ev_cls=np.array([self.cls.get(ev.cls, -1) for ev in events], np.int16),
            ev_t=np.array([[ev.t_lock, ev.t_away, ev.t_ret, ev.t_end]
                           for ev in events], np.int32).reshape(-1, 4),
            ev_kind=np.array([ev.kind for ev in events], np.int8),
            ev_plan=np.array([[ev.plan_T, ev.plan_deg] for ev in events],
                             np.float32).reshape(-1, 2),
            class_names=json.dumps({v: k for k, v in self.cls.items()}),
            h=H, w=W, recipe=json.dumps(self.recipe))
        print(f"  [shard] {self.name}: {self.n} frames sealed", flush=True)


