"""
train_text — THE blessed text-pretrain orchestrator (on the assembled core).

A first-class module of the text modality — peer to tokenizer.py / data_source.py /
evaluator.py, NOT a throwaway sample. It is the maintained, importable Orchestrator
you may run as-is or copy-and-adapt (Library over Framework, Examples
over Interfaces; an Orchestrator is itself a reusable building block). This is the
assembly-world reference. What it shows:

  1. Modality manifests -> ONE shared vocab. Each mounted modality declares
     {name, type_id, vocab_size, local-ID producer}; the assembler stacks the
     bands into a VocabLayout. vocab_size / n_token_types are FACTS OF THE
     ASSEMBLY — never config constants that could disagree with the artifact.
  2. TWO system objects come out of assembly and ride the shared bag,
     explicitly passed, never global: `layout` (structure: pure integers) and
     `control_resolver` (protocol: name -> global id, THE authority on
     control-token names).
  3. Orchestrators COMPOSE: source fragments (SOURCE_TYPES) from each mounted
     modality; evaluators into the Trainer's list; recipes are experiment
     property (TextDataSource defaults to text's own single-modality recipe).
  4. The assembly-time lock: every protocol name must resolve to the trained
     artifact's id — registry/artifact drift fails loudly at startup.

Usage (module entry, runnable from anywhere — the editable install resolves it):

    python -m modalities.text.train_text
    python -m modalities.text.train_text model.depth=12 max_steps=100
    python -m modalities.text.train_text use_compile=false wandb.enabled=true
    torchrun --nproc_per_node=2 --standalone -m modalities.text.train_text

Config: modalities/text/configs/train_text.yaml (co-located; inherits mechanism
defaults from core/configs/train_base.yaml). Defaults are the known-good
settings from the MFU audit — see the yaml for the depth->MFU expectations.
"""

import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

# ${eval:'...'} — a tiny arithmetic resolver so the config can hold DERIVATION
# RULES, not just literals: model.dim = 64·depth lives IN train_text.yaml,
# visibly, and a CLI override (model.dim=1024) simply replaces the rule.
# Restricted eval: arithmetic + max/min, no builtins.
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver(
        "eval", lambda expr: eval(expr, {"__builtins__": {}}, {"max": max, "min": min}))

from core.training.model_setup import build_system, print0
from core.training.trainer import Trainer, create_optimizers
from core.model.gpt import GPT, GPTConfig  # WHICH model = this orchestrator's decision
from core.data.mixed_dataloader import MixedDataLoader

import modalities.text
import modalities.control
from modalities.assembler import build_layout
from modalities.control import CONTROL_TOKENS, display_form, make_control_resolver
from modalities.text import TextEvaluator, get_tokenizer

# The orchestrator composes the source fragments of the mounted modalities.
SOURCE_TYPES = {
    **modalities.text.SOURCE_TYPES,
}


def resolve_trunk(path):
    """The trunk class this orchestrator builds. Default: the reference GPT.
    A config `model.trunk_class: pkg.mod.ClassName` names any class satisfying the
    trunk contract (see core/model/gpt.py) — so a project can drive an architecture
    variant through this same orchestrator without editing core or forking here."""
    if not path:
        return GPT
    module_name, _, cls_name = str(path).rpartition(".")
    return getattr(importlib.import_module(module_name), cls_name)


def assemble_vocab(tokenizer):
    """[text, control] -> (layout, control_resolver).

    Pinned to the trained artifact's geometry: {text:0, control:2},
    n_token_types=3 (canonical ids leave room for motion at type 1 — the
    layout allows gaps, and a [text, control] assembly stays shape-identical
    to the three-modality world)."""
    text = modalities.text.manifest(tokenizer)
    control = modalities.control.manifest()
    assert (text.type_id, control.type_id) == (0, 2), "canonical type ids moved"

    layout = build_layout([text, control])
    assert layout.n_token_types == 3, f"expected n_token_types=3, got {layout.n_token_types}"
    assert layout.vocab_size == tokenizer.get_vocab_size(), "layout/artifact vocab mismatch"

    resolver = make_control_resolver(control, layout)
    # Protocol/artifact lock: the registry's names must land exactly on the
    # trained artifact's special ids (catches registry<->pkl drift).
    for name in CONTROL_TOKENS:
        artifact_id = tokenizer.encode_special(display_form(name))
        assert resolver.resolve(name) == artifact_id, \
            f"protocol drift: {name} -> {resolver.resolve(name)} != artifact {artifact_id}"
    assert resolver.resolve("not_a_protocol_name") is None

    return layout, resolver


@hydra.main(version_base=None, config_path="configs", config_name="train_text")
def main(cfg: DictConfig) -> None:
    print0("=" * 80)
    print0("train_text — text pretraining on the assembled core (modalities/text)")
    print0("=" * 80)

    config = OmegaConf.to_container(cfg, resolve=True)

    sequence_len = config['sequence_len']
    device_batch_size = config['device_batch_size']
    model_config = config['model']
    optimizer_config = config['optimizer']

    # --- Assemble the shared vocab from modality manifests ---
    print0("\nLoading tokenizer + assembling VocabLayout from modalities/...")
    tokenizer = get_tokenizer()
    layout, control_resolver = assemble_vocab(tokenizer)
    print0(f"  layout ranges={dict(layout.ranges)} -> vocab_size={layout.vocab_size}, "
           f"n_token_types={layout.n_token_types}")

    # --- Model (trunk config; vocab facts come FROM the layout, geometry facts
    #     come FROM the config — the derivation rules live in train_text.yaml) ---
    depth = model_config['depth']
    model_dim = model_config['dim']
    num_heads = model_config['n_head']
    gpt_config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=layout.vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=model_config['n_kv_head'],
        n_embd=model_dim,
        n_token_types=layout.n_token_types,
    )
    # WHICH trunk to build is the config's choice — default is the reference GPT,
    # but `model.trunk_class` names any class satisfying the trunk contract (e.g. a
    # project's architecture ablation). The orchestrator honors core's pluggable-
    # trunk seam rather than hardcoding one model.
    trunk_cls = resolve_trunk(model_config.get('trunk_class'))
    print0(f"\nAssembling d{depth} {trunk_cls.__name__} (dim={model_dim}, heads={num_heads}, "
           f"vocab={layout.vocab_size})...")
    setup = build_system(trunk_cls, gpt_config, use_compile=config.get('use_compile', True),
                                                seed=config.get('seed', 42))
    system = setup['system']
    rank = setup['rank']
    world_size = setup['world_size']

    # --- Data: the TWO system objects (layout + resolver) ride the shared bag ---
    data_config = config['data']
    device = 'cuda'
    sources = []
    for sc in data_config['sources']:
        sc = dict(sc)
        sc.setdefault('sequence_len', sequence_len)
        sc.setdefault('device', device)
        sources.append(sc)

    loader_config = {
        'batch_size': device_batch_size,
        'data': {
            'sequence_len': sequence_len,
            'sources': sources,
        },
    }
    print0(f"\nBuilding MixedDataLoader ({len(sources)} source(s))...")
    dataloader = MixedDataLoader(
        loader_config=loader_config,
        tokenizers={'text': tokenizer, 'layout': layout, 'control_resolver': control_resolver},
        source_types=SOURCE_TYPES,
        resume_state_dict=None,
    )

    # --- Optimizers (built from the system, grouped by trunk/head) ---
    print0("Creating optimizers...")
    optimizers = create_optimizers(system, optimizer_config, world_size=world_size)

    # --- Evaluators (each modality's own, injected) ---
    eval_config = config.get('evaluation', {})
    evaluators = []
    text_eval = eval_config.get('text', {})
    if text_eval.get('enabled', True):
        evaluators.append(TextEvaluator(text_eval, device_batch_size, sequence_len))

    # --- Trainer ---
    print0("Creating trainer...")
    trainer = Trainer(
        system=system,
        optimizers=optimizers,
        dataloader=dataloader,
        config=config,
        rank=rank,
        world_size=world_size,
        debug_tokenizer=tokenizer,
        evaluators=evaluators,
    )

    print0("Starting training...\n")
    trainer.train()
    print0("\n" + "=" * 80)
    print0("✓ train_text completed")
    print0("=" * 80)


if __name__ == "__main__":
    main()
