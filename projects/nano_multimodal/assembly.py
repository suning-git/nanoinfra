"""
assembly.py — line name -> everything a run needs. The ONE path into the data.

WHY THIS IS NOT INSIDE train.py. Orchestrators own assembly (core's design), and
train.py IS the orchestrator — but assembly has a SECOND legitimate consumer: the
data browser (serve/browse.py), which must show the batches training actually
sees. If the browser assembled its own loader, the two would be free to drift, and
this repo has already paid for exactly that bug once: training assembled sequences
through a recipe while evaluation packed raw tokens, the two rulers disagreed for
months, and nothing in the log ever said so (modalities/text/data_source.py
documents the post-mortem). One function, two callers, no room to disagree.

So: train.py is still the orchestrator (it decides WHAT to run and wires the
Trainer); this module is the assembly it owns, factored out so the browser can
walk the identical path.

    vocab = assemble_vocab("video", config)          # cheap, no CUDA
    loader = build_loader("video", config, vocab, "train", device)

Read build_layout_for_line() first — it is where the project's one idea lives.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from projects.nano_multimodal import spec

spec.pin_tokenizer()                                         # MUST precede modalities.text

import modalities.control                                    # noqa: E402
import modalities.text                                       # noqa: E402
from modalities.assembler import Modality, build_layout      # noqa: E402
from modalities.control import CONTROL_TOKENS, make_control_resolver  # noqa: E402

from core.data.batchers import PaddingBatcher                # noqa: E402
from core.data.mixed_dataloader import MixedDataLoader       # noqa: E402
from core.data.supervision import build_NTP_supervised_sample  # noqa: E402
from core.model.gpt import GPTConfig                         # noqa: E402
from core.tokenization.vocab_layout import VocabLayout       # noqa: E402
from core.utils import print0                                # noqa: E402

class AliasResolver:
    """The control resolver, plus this project's readable delimiter names.

    `video_start` is really `ctrl0` — one of the four unnamed slots the control
    registry reserves so a project can claim one without editing core. That is the
    right mechanism, but `ctrl0` in a row template is a name a student has to look
    up. This wrapper lets the yaml say `video_start` while the protocol underneath
    stays exactly what core shipped.

    One way, and local: the registry is untouched, checkpoints do not know these
    names exist, and an unknown name still resolves to None (which is how
    SequenceRecipe tells a control token from a data field).
    """

    def __init__(self, inner, aliases: Dict[str, str]):
        self._inner = inner
        self._aliases = dict(aliases)

    def resolve(self, name: str):
        return self._inner.resolve(self._aliases.get(name, name))

    def name_of(self, token_id: int):
        """global id -> the readable name, for the browser's row legend. Prefers the
        alias, so a video row's delimiters read `video_start`, not `ctrl0`."""
        rev = {v: k for k, v in self._aliases.items()}
        for canonical in CONTROL_TOKENS:
            if self._inner.resolve(canonical) == token_id:
                return rev.get(canonical, canonical)
        return None


def register_resolvers():
    """The two custom ${...} resolvers this project's configs use.

    Registered HERE rather than in train.py because train.py is not the only reader:
    the browser, the tests and scaling.py all load a config without going through
    hydra, and a resolver that only exists inside the CLI entry point makes those
    configs unreadable everywhere else — which is how a "config" quietly becomes
    "whatever the trainer happens to parse".

      ${eval:'...'}   restricted arithmetic, so a config can hold DERIVATION RULES
                      (model.dim = 64*depth stays visible in the yaml, and a CLI
                      override replaces the rule). No builtins.
      ${repo:a/b}     an absolute path under the repo root. Replaces
                      ${hydra:runtime.cwd}, which resolves only under hydra.
    """
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver(
            "eval", lambda expr: eval(expr, {"__builtins__": {}}, {"max": max, "min": min}))
    if not OmegaConf.has_resolver("repo"):
        OmegaConf.register_new_resolver("repo", lambda *p: str(Path(spec.REPO).joinpath(*p)))


def load_config(line: str, **overrides) -> Dict[str, Any]:
    """A line's config, resolved, WITHOUT hydra — for the browser, the tests and
    scaling.py. Composition (base.yaml + the line config) is done by hand here
    because hydra's defaults list is a hydra feature; the merge order is the same."""
    register_resolvers()
    base = OmegaConf.load(spec.PROJECT / "configs" / "base.yaml")
    line_cfg = OmegaConf.load(spec.PROJECT / "configs" / f"{line}.yaml")
    from core.utils import get_base_dir  # noqa: F401  (train_base uses oc.env)
    core_base = OmegaConf.load(Path(spec.REPO) / "core" / "configs" / "train_base.yaml")
    cfg = OmegaConf.merge(core_base, base, line_cfg, OmegaConf.create(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


TYPE_IDS = {
    "text": spec.TEXT_TYPE_ID,
    "motion": spec.MOTION_TYPE_ID,
    "control": spec.CONTROL_TYPE_ID,
    "video": spec.VIDEO_TYPE_ID,
    "action": spec.ACTION_TYPE_ID,
}


@dataclass
class Vocab:
    """The assembled vocabulary for one line. Cheap to build (no CUDA, no codec)."""
    line: str
    layout: VocabLayout                # this line's COMPACT layout — what the model sees
    canonical: VocabLayout             # the full five-band table — display only
    resolver: Any                      # control name -> global id, in THIS line's layout
    tokenizers: Dict[str, Any]         # the shared bag every DataSource receives
    widths: Dict[str, int]
    sequence_len: int                  # T, positions per row; a source sample is up to T + 1 raw tokens

    @property
    def active(self) -> List[str]:
        return spec.LINES[self.line]

    def describe(self) -> str:
        bands = ", ".join(f"{n}[{self.layout.offset(TYPE_IDS[n])}"
                          f"..{self.layout.offset(TYPE_IDS[n]) + self.widths[n]})"
                          for n in spec.BAND_ORDER if n in self.active)
        return (f"line={self.line} | bands: {bands} | vocab_size={self.layout.vocab_size} "
                f"| n_token_types={self.layout.n_token_types} | seq_len={self.sequence_len}")


# --------------------------------------------------------------------------
# band widths
# --------------------------------------------------------------------------

def band_widths(tokenizer=None, motion_vocab=None, video_vocab=None) -> Dict[str, int]:
    """Every band's width.

    Read off the ARTIFACT wherever one is in hand, and asserted against spec.py's
    declaration every time. The declaration is what lets a line draw bands it does
    not activate (the browser's vocab panorama needs the whole table); the assert
    is what keeps declaration and artifact from drifting apart.

    Note where the motion and video widths come from: not from the codec, but from
    the CACHE SIDECAR — the artifact that actually produced the data on disk. A
    cache encoded by one codec and decoded by another is integers either way, and
    nothing complains; modalities/motion/tokenizers/REGISTRY.md records that this
    has happened. Asking the cache closes that hole without loading 73MB of codec
    weights to read one number.
    """
    widths = {
        "text": spec.TEXT_VOCAB,
        "motion": spec.MOTION_VOCAB,
        "control": len(CONTROL_TOKENS),
        "video": spec.VIDEO_VOCAB,
        "action": spec.N_ACTIONS,
    }
    if tokenizer is not None:
        actual = tokenizer.get_vocab_size() - len(tokenizer.get_special_tokens())
        assert actual == spec.TEXT_VOCAB, (
            f"tokenizer artifact carries {actual} content ids but spec.TEXT_VOCAB says "
            f"{spec.TEXT_VOCAB}. Every band after text moves with this number — a "
            f"checkpoint trained on one cannot be loaded against the other. Retrain the "
            f"tokenizer at vocab 32768, or update spec and accept that old checkpoints die.")
        widths["text"] = actual
    if motion_vocab is not None:
        assert motion_vocab == spec.MOTION_VOCAB, (
            f"the motion cache was encoded with a {motion_vocab}-code codec but "
            f"spec.MOTION_VOCAB says {spec.MOTION_VOCAB} — the band would be the wrong "
            f"width and motion tokens would land on top of whatever follows.")
        widths["motion"] = motion_vocab
    if video_vocab is not None:
        assert video_vocab == spec.VIDEO_VOCAB, (
            f"the video cache was encoded with a {video_vocab}-code codec but "
            f"spec.VIDEO_VOCAB says {spec.VIDEO_VOCAB}.")
        widths["video"] = video_vocab
    return widths


def _modality(name: str, widths: Dict[str, int]) -> Modality:
    """One band's manifest. `control` brings its name table (the resolver is built
    from it); the others need only a width, because training never asks a codec for
    anything but how many codes exist."""
    if name == "control":
        m = modalities.control.manifest()
        assert m.vocab_size == widths["control"]
        return m
    return Modality(name=name, type_id=TYPE_IDS[name], vocab_size=widths[name])


def build_layout_for_line(line: str, widths: Dict[str, int]):
    """★ THE ONE IDEA. Stack this line's ACTIVE bands, in spec.BAND_ORDER, into a
    compact VocabLayout; also build the canonical five-band table for display.

    Returns (compact, canonical, resolver).

    Two consequences, both visible in the browser:

      * vocab_size is the sum of the ACTIVE bands. The video line is 64037, not
        97299 — it never carries a text band it cannot emit. Measured on this box:
        one dead 32750-wide band costs 1.43x the step time (289 -> 202 ms/step).
      * Token IDs are therefore per-line: `control` starts at 32750 on the text
        line and at 0 on the video line. Token TYPE ids are NOT — a type 4 token is
        a video token everywhere. That is what lets one colour scheme serve all
        three lines.
    """
    active = spec.LINES[line]
    assert "control" in active, f"line {line!r} must activate the control band (delimiters)"
    unknown = set(active) - set(spec.BAND_ORDER)
    assert not unknown, f"line {line!r} names unknown bands: {sorted(unknown)}"

    ordered = [n for n in spec.BAND_ORDER if n in active]
    compact = build_layout([_modality(n, widths) for n in ordered])
    canonical = build_layout([_modality(n, widths) for n in spec.BAND_ORDER])
    resolver = AliasResolver(make_control_resolver(_modality("control", widths), compact),
                             spec.CONTROL_ALIASES)

    # Protocol lock: every delimiter this project's recipes name must resolve. A
    # band offset that is wrong by one trains perfectly and decodes into nonsense,
    # so it has to fail here, loudly, before the first forward.
    for name in ("bos", "eos", "video_start", "video_end",
                 spec.MOTION_START, spec.MOTION_END, "text_start", "text_end"):
        assert resolver.resolve(name) is not None, f"control slot {name!r} vanished"
    assert resolver.resolve("not_a_control_name") is None

    expected = sum(widths[n] for n in ordered)
    assert compact.vocab_size == expected, (
        f"assembled vocab {compact.vocab_size} != sum of active bands {expected}")
    return compact, canonical, resolver


# --------------------------------------------------------------------------
# cache sidecars — where the motion / video band widths really come from
# --------------------------------------------------------------------------

def motion_cache_facts(source_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Which codec made this motion cache, and how wide its band is.

    THE TAG IS NOT ENOUGH, and we know that empirically. Every cache in
    outputs/motion_caches that carries a sidecar tags itself `rot139_kin_fsq2`,
    and only the bones-family ones actually decode to an upright human through
    that codec (measured: mean per-frame body height 1.48 m for bones against
    0.98-1.13 m for the others). The codec is fine — a bones_seed ground-truth
    clip round-trips at 1.47 -> 1.47 m. Something ELSE differs about how the other
    caches were prepared before encoding, and the sidecar has no field for it (note
    that the bones code files are named `..._canon.npz` and the others are not).

    So the tag is checked where it exists, the config may declare it where it does
    not, and neither is trusted on its own: tests/test_motion_cache.py decodes real
    rows and asserts the body stands up. That check is what actually rules out the
    failure REGISTRY.md warns about, because it looks at the poses rather than at a
    string.
    """
    import json
    cache_name = source_cfg["cache"]
    declared = source_cfg.get("codec")
    path = spec.MOTION_CACHE_DIR / cache_name.replace(".npz", ".json")
    facts = json.loads(path.read_text()) if path.exists() else {}

    tag = declared if declared is not None else facts.get("codec")
    if tag is None:
        raise ValueError(
            f"{cache_name} has no sidecar and the source declares no `codec:` — say "
            f"which codec made it. Codes from two different codecs are the same "
            f"integers on disk and look identical (REGISTRY.md), so this cannot be "
            f"inferred.")
    if declared is not None and facts.get("codec") not in (None, declared):
        # The DECLARATION wins, and says so every time. A sidecar can be wrong —
        # t2m_humanml3d_*.json names rot139_kin_fsq2 and decodes to a 1.01 m body
        # through it, against 1.38 m through rot139_vqvae — but overriding recorded
        # provenance must never be quiet, or the next person inherits a mystery
        # instead of a decision.
        tag = declared
        print(f"  NOTE {cache_name}: sidecar says codec {facts['codec']!r}, config "
              f"declares {declared!r} — using the config. The check that settles it "
              f"is tests/test_motion_cache.py, which decodes rows and looks at the "
              f"body, not at the tag.")
    if tag != spec.MOTION_CODEC:
        raise ValueError(
            f"{cache_name} was made by codec {tag!r} but spec.MOTION_CODEC is "
            f"{spec.MOTION_CODEC!r}. Decoding these codes with the wrong codec "
            f"produces plausible-looking nonsense.")
    return {"codec": tag, "vocab_size": facts.get("vocab_size", spec.MOTION_VOCAB),
            "downsample": facts.get("downsample", spec.MOTION_DOWNSAMPLE),
            # WHICH authority supplied the codec name, so the log never implies the
            # sidecar agreed when it was overridden.
            "source": "config" if declared is not None else "sidecar",
            "has_sidecar": path.exists()}


def video_cache_facts(cache_dir) -> Dict[str, Any]:
    """Read the video cache's meta.json: the shape contract it was built for and
    the codec that made it. Asserted against spec.video_shape() by the caller."""
    import json
    from pathlib import Path
    path = Path(cache_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — build the video cache first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def assemble_vocab(line: str, config: Dict[str, Any]) -> Vocab:
    """Vocabulary + sequence length for a line. No CUDA, no model, no codec —
    cheap enough for the browser to call on every request."""
    active = spec.LINES[line]
    tokenizer = modalities.text.get_tokenizer() if "text" in active else None

    motion_vocab = video_vocab = None
    if "motion" in active:
        motion_vocab = motion_cache_facts(_first_source(config))["vocab_size"]
    if "video" in active:
        meta = video_cache_facts(_first_source(config)["cache"])
        video_vocab = spec.VIDEO_VOCAB if meta.get("codec") == spec.CODEC_NAME else None
        assert meta.get("codec") == spec.CODEC_NAME, (
            f"video cache was made by {meta.get('codec')!r}, spec says {spec.CODEC_NAME!r}")
        shape = spec.video_shape(config["clip"]["frames"], config["clip"]["res"])
        cached = meta.get("shape_contract", {})
        for k in ("codes_per_frame", "n_latent", "td"):
            assert cached.get(k) == shape[k], (
                f"cache shape_contract[{k}]={cached.get(k)} != run {shape[k]} — rebuild "
                f"the cache for this clip length")

    widths = band_widths(tokenizer, motion_vocab, video_vocab)
    layout, canonical, resolver = build_layout_for_line(line, widths)

    # Artifact lock. On any line that carries text, control follows it directly
    # (spec.BAND_ORDER), so the assembled ids for the control band must be exactly
    # the special-token ids the BPE artifact assigns. Checking it here catches
    # registry-vs-pkl drift at startup instead of at decode time, where it looks
    # like a bad model rather than a bad offset.
    if tokenizer is not None:
        from modalities.control import display_form
        for name in CONTROL_TOKENS:
            assembled = resolver.resolve(name)
            artifact = tokenizer.encode_special(display_form(name))
            assert assembled == artifact, (
                f"protocol drift: control token {name!r} is id {assembled} in this "
                f"assembly but {artifact} in the tokenizer artifact")

    seq = config.get("sequence_len")
    if seq is None:
        assert line == "video", "only the video line derives its sequence_len"
        # a video row of row_len tokens trains as row_len - 1 positions (the next-token shift)
        seq = spec.video_shape(config["clip"]["frames"], config["clip"]["res"])["row_len"] - 1

    tokenizers: Dict[str, Any] = {"layout": layout, "control_resolver": resolver}
    if tokenizer is not None:
        tokenizers["text"] = tokenizer

    return Vocab(line=line, layout=layout, canonical=canonical, resolver=resolver,
                 tokenizers=tokenizers, widths=widths, sequence_len=seq)


def _first_source(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["data"]["sources"][0]


def resolve_sources(config, vocab: Vocab, split: str, device: str,
                    rank: int = 0, world_size: int = 1) -> List[Dict[str, Any]]:
    """Source DECLARATIONS -> concrete source configs.

    Turns the recipe NAME into the recipe dict, fills in per-run facts
    (sample_len = sequence_len + 1 raw tokens, or max_sample_len for t2m; device, rank), and resolves the
    text line's file list through modalities.text.datasets — where splits are declared, never inferred from a
    directory listing.

    `split` picks between the config's `sources` (train) and `val_sources` (val).
    Both are DECLARED in yaml rather than derived by string substitution: a val set
    that is guessed at is a ruler nobody audited.
    """
    key = "sources" if split == "train" else "val_sources"
    declared = config["data"].get(key)
    if not declared:
        raise KeyError(
            f"data.{key} is not declared in the config for line {vocab.line!r}. "
            f"Both the training source and its validation counterpart are written "
            f"down; neither is inferred.")

    recipes = config["data"].get("recipes") or {}
    out = []
    for sc in declared:
        sc = dict(sc)
        name = sc.get("recipe")
        if name not in recipes:
            raise KeyError(f"unknown recipe {name!r}; declared: {sorted(recipes)}")
        sc["recipe_name"] = name
        sc["recipe"] = recipes[name]
        length_key = "max_sample_len" if sc["type"] == "t2m" else "sample_len"
        if "sequence_len" in sc:
            raise KeyError(f"a {sc['type']} source declares `sequence_len`; it takes `{length_key}` (raw tokens "
                           f"{'a pair may take' if sc['type'] == 't2m' else 'per sample'}, default the line's "
                           f"sequence_len + 1)")
        sc.setdefault(length_key, vocab.sequence_len + 1)
        sc.setdefault("device", "cpu")     # sources yield CPU samples; the loader moves whole batches
        sc.setdefault("seed", config.get("seed", spec.SEED))
        sc.setdefault("rank", rank)
        sc.setdefault("world_size", world_size)
        if sc["type"] == "text" and "files" not in sc:
            from modalities.text.datasets import resolve_split
            sc["files"] = resolve_split(config["data"], sc["dataset"], sc["split"])
        out.append(sc)
    return out


def build_loader(line: str, config: Dict[str, Any], vocab: Vocab, split: str,
                 device: str, batch_size: int, rank: int = 0, world_size: int = 1):
    """The MixedDataLoader for one split. THE only way this project reads data —
    train.py and serve/browse.py both come through here."""
    from projects.nano_multimodal.sources import source_types

    from modalities.text.streams import ntp_pad_values

    sources = resolve_sources(config, vocab, split, device, rank, world_size)
    loader_config = {"data": {"sources": sources}}
    # Each raw sample is prepared for next-token prediction (L + 1 tokens -> L positions,
    # unsupervised tokens -> IGNORE targets), one sample per row (PaddingBatcher,
    # batch_size rows of sequence_len positions, padding with eos and IGNORE), and the
    # batch is moved to `device` in one copy per field.
    return MixedDataLoader.from_config(
        loader_config, vocab.tokenizers, source_types(),
        build_supervised_sample_fn=build_NTP_supervised_sample,
        create_batcher_fn=partial(PaddingBatcher, shape=(batch_size, vocab.sequence_len),
                                  pad_values=ntp_pad_values(vocab.layout, vocab.resolver)),
        device=device)


def gpt_config_for(config: Dict[str, Any], vocab: Vocab) -> GPTConfig:
    """Trunk geometry from the config; vocab facts from the ASSEMBLY.

    vocab_size and n_token_types are deliberately absent from every yaml in this
    project: a config constant can disagree with the artifact, and a band offset
    that is wrong by one trains perfectly and decodes into nonsense.
    """
    m = config["model"]
    return GPTConfig(
        sequence_len=vocab.sequence_len,
        vocab_size=vocab.layout.vocab_size,
        n_layer=m["depth"],
        n_head=m["n_head"],
        n_kv_head=m["n_kv_head"],
        n_embd=m["dim"],
        n_token_types=vocab.layout.n_token_types,
    )


def assemble(line: str, config: Dict[str, Any], split: str = "train",
             device: str = "cuda", batch_size: int = 8):
    """Vocabulary + loader in one call. What serve/browse.py uses."""
    vocab = assemble_vocab(line, config)
    loader = build_loader(line, config, vocab, split, device, batch_size)
    return vocab, loader
