"""
convae.train — the shared training harness for the conv-AE tokenizer family.

The final recipe, crystallized: train a MotionVQVAE / MotionFSQ2
on rot139 windows with an optional kinematic loss (FK position + velocity +
foot-contact), select best-val (root-rel), and evaluate on the FAIR
global-canonical metric. Composes entirely on modalities/motion/data — no
research-project imports; this is the reproduction path a shelf tokenizer's
train() calls.

A tokenizer folder passes its recipe (a dict, usually from recipe.yaml); the
ONLY per-tokenizer differences are the quantizer + dims + objective weights +
data, all in the recipe. `train_codec(recipe, out_path)` reproduces the artifact.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from modalities.motion.data import dataset as md
from modalities.motion.data import fk_torch
from modalities.motion.data.converters import smpl_body as B
from modalities.motion.data.converters import smpl_to_rot139 as conv
from modalities.motion.tokenizers._convae.codec import save_checkpoint
from modalities.motion.tokenizers._convae.models import MotionFSQ2, MotionVQVAE

FEET = [7, 10, 8, 11]   # SMPL-22 foot joints (contacts 135:139)
TN = 64                 # training/eval window length (frames)

DEFAULTS = dict(
    rep="rot139", quantizer="fsq2", d_feat=139, code_dim=512, width=512, downsample=4,
    levels=[8, 8, 8], n_codes=512, commit_weight=0.25,
    lam=100.0, lr=2e-4, bs=256, steps=30000, max_windows=500000,
    data="bones_seed", seed=0, eval_every=2000,
)


def _raw_windows(clips, win, stride):
    """Un-normalized windows (for the fair-eval set) — the data leg's make_windows
    normalizes; here we need raw features to FK against the truth."""
    out = []
    for c in clips:
        for s in range(0, len(c) - win + 1, stride):
            out.append(c[s:s + win])
    return np.stack(out).astype(np.float32)


def _facing(gp):
    a = (gp[:, 2, :2] - gp[:, 1, :2]) + (gp[:, 17, :2] - gp[:, 16, :2])
    return np.unwrap(np.arctan2(a[:, 0], -a[:, 1]))


def _canon(gp):
    th0 = _facing(gp)[0]
    c, s = np.cos(-th0), np.sin(-th0)
    R = np.array([[c, -s], [s, c]])
    out = gp.copy(); out[..., :2] = (gp[..., :2] - gp[:1, 0:1, :2]) @ R.T
    return out


def _build_model(r):
    if r["quantizer"] == "vq":
        m = MotionVQVAE(d_feat=r["d_feat"], n_codes=r["n_codes"], code_dim=r["code_dim"],
                        width=r["width"], commit_weight=r.get("commit_weight", 0.25),
                        downsample=r["downsample"])
        cfg = {"rep": r["rep"], "d_feat": r["d_feat"], "n_codes": r["n_codes"],
               "code_dim": r["code_dim"], "width": r["width"],
               "commit_weight": r.get("commit_weight", 0.25),
               "downsample": r["downsample"], "quantizer": "vq"}
    else:
        m = MotionFSQ2(r["d_feat"], width=r["width"], code_dim=r["code_dim"],
                       downsample=r["downsample"], levels=list(r["levels"]))
        cfg = {"rep": r["rep"], "d_feat": r["d_feat"], "width": r["width"],
               "code_dim": r["code_dim"], "downsample": r["downsample"],
               "levels": list(r["levels"]), "quantizer": "fsq2"}
    return m, cfg


def train_codec(recipe: dict, out_path: str, device: str = "cuda", progress=print) -> dict:
    """Reproduce a conv-AE tokenizer from a recipe dict. Saves a MotionCodec-loadable
    checkpoint to out_path; returns {fair_global_mpjpe_cm, heading_deg, best_rootrel_cm}."""
    r = {**DEFAULTS, **recipe}
    torch.manual_seed(r["seed"])
    J, parents = B.load_body_model("neutral")

    # ---- data (via the crystallized data leg) --------------------------------
    tr_clips, _ = md.load_or_build("train", r["data"], verbose=False)
    va_clips, _ = md.load_or_build("val", r["data"], verbose=False)
    D = tr_clips[0].shape[1]
    norm = md.Normalizer.fit(tr_clips)
    tr = md.make_windows(tr_clips, TN, TN // 2, norm)
    va = md.make_windows(va_clips, TN, TN // 2, norm)
    mw = r["max_windows"]
    if mw and len(tr) > mw:
        tr = tr[np.random.default_rng(r["seed"]).permutation(len(tr))[:mw]]
    if mw and len(va) > mw // 4:
        va = va[np.random.default_rng(r["seed"] + 1).permutation(len(va))[:mw // 4]]
    Xt = torch.from_numpy(np.ascontiguousarray(tr, dtype=np.float32)); del tr
    Xv = torch.from_numpy(np.ascontiguousarray(va, dtype=np.float32)); del va
    Wfair = _raw_windows(va_clips, TN, TN)
    Wfair = Wfair[np.random.default_rng(0).permutation(len(Wfair))[:300]]
    del tr_clips, va_clips
    mean_t = torch.tensor(norm.mean, device=device)
    std_t = torch.tensor(norm.std, device=device)

    model, cfg = _build_model({**r, "d_feat": D})
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    progress(f"[{r['quantizer']}] train {tuple(Xt.shape)} val {tuple(Xv.shape)} "
             f"fair {len(Wfair)} | params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def unnorm(f):
        return f * std_t + mean_t

    def kin_terms(recon, target):
        rf, tf = unnorm(recon), unnorm(target)
        pr, pt = fk_torch.features_to_rootrel(rf), fk_torch.features_to_rootrel(tf)
        pos = F.mse_loss(pr, pt)
        vel = F.mse_loss(pr[:, 1:] - pr[:, :-1], pt[:, 1:] - pt[:, :-1])
        dfoot = pr[:, 1:, FEET] - pr[:, :-1, FEET]
        disp = rf[..., 132:134]; h = rf[..., 134]
        inc = torch.cat([disp[:, 1:], (h[:, 1:] - h[:, :-1])[..., None]], -1)
        wfv = dfoot + inc[:, :, None, :]
        contact = (tf[..., 135:139] > 0.5).float()
        planted = contact[:, 1:] * contact[:, :-1]
        foot = (planted * (wfv ** 2).sum(-1)).sum() / planted.sum().clamp(min=1.0)
        return pos, vel, foot

    @torch.no_grad()
    def val_rootrel():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, min(len(Xv), 8192), 512):
            xb = Xv[i:i + 512].to(device)
            rec = model.decode(model.encode(xb))
            pr = fk_torch.features_to_rootrel(unnorm(rec))
            pt = fk_torch.features_to_rootrel(unnorm(xb))
            tot += (pr - pt).norm(dim=-1).mean().item() * len(xb); n += len(xb)
        model.train()
        return tot / n * 100

    # ---- train (best-val root-rel selection) ---------------------------------
    rng = np.random.default_rng(r["seed"]); model.train()
    step, best, best_state, lam = 0, float("inf"), None, r["lam"]
    while step < r["steps"]:
        idx = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - r["bs"] + 1, r["bs"]):
            xb = Xt[idx[i:i + r["bs"]]].to(device)
            out = model(xb)
            loss = out["loss"]
            if lam:
                pos, vel, foot = kin_terms(out["recon"], xb)
                loss = loss + lam * (pos + vel + foot)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % r["eval_every"] == 0 or step == r["steps"]:
                cm = val_rootrel(); tag = ""
                if cm < best:
                    best, best_state = cm, copy.deepcopy(model.state_dict()); tag = " *best*"
                progress(f"[{r['quantizer']}] step {step:6d} loss {loss.item():.3f} "
                         f"ppl {out['perplexity']:.0f} | root-rel {cm:.2f}cm{tag}")
            if step >= r["steps"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- fair global-canonical eval (the published protocol) -----------------
    model.eval(); gmp, herr = [], []
    with torch.no_grad():
        for w in Wfair:
            x = torch.tensor(((w - norm.mean) / norm.std)[None], dtype=torch.float32, device=device)
            rec = (model.decode(model.encode(x))[0] * std_t + mean_t).cpu().numpy()
            L = min(len(rec), TN)
            lRr, trr = conv.features_to_smpl(rec[:L], np.zeros(3)); gpr, _ = B.smpl_fk(lRr, trr, J, parents)
            lRt, trt = conv.features_to_smpl(w[:L], np.zeros(3)); gpt, _ = B.smpl_fk(lRt, trt, J, parents)
            gpr, gpt = _canon(gpr), _canon(gpt)
            gmp.append(np.linalg.norm(gpr - gpt, axis=-1).mean())
            he = np.abs((_facing(gpr) - _facing(gpr)[0]) - (_facing(gpt) - _facing(gpt)[0]))
            herr.append(np.degrees(he).mean())
    fair_cm, fair_deg = float(np.mean(gmp)) * 100, float(np.mean(herr))
    metrics = {"fair_global_mpjpe_cm": fair_cm, "heading_deg": fair_deg, "best_rootrel_cm": best}
    progress(f"[{r['quantizer']}] FAIR global {fair_cm:.2f} cm | heading {fair_deg:.1f} deg | "
             f"best root-rel {best:.2f} cm")

    cfg["train"] = {k: r[k] for k in ("steps", "bs", "lr", "lam", "max_windows", "data", "seed")}
    cfg["train"]["selection"] = "best val root-rel"
    cfg["fair_eval"] = {**metrics, "protocol": "fair global-canonical N=300 stride-64 rng(0)"}
    save_checkpoint(model, cfg, norm.mean, norm.std, out_path)
    progress(f"saved -> {out_path}")
    return metrics
