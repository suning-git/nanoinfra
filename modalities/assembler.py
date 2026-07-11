"""
The assembler — shared assembly infrastructure for all modalities.

A Modality is a declarative MANIFEST (a registration form), not a behavior
interface: it declares "my name, my type identity, my local band size, my
local-ID producer". The assembler stacks the declared bands into one shared
integer space (a core VocabLayout). Behavior lives in each modality's own
folder; core provides only the mechanism (VocabLayout).

Not core (core stays zero-modality), not per-experiment (every orchestrator
needs it) — shared infra, living beside the modality folders.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from core.tokenization.vocab_layout import VocabLayout

# Bag keys the orchestrator reserves for the two system-level assembly products
# (structure + protocol). A modality may not claim them as its name.
RESERVED_BAG_KEYS = frozenset({"layout", "control_resolver"})


@dataclass
class Modality:
    """Assembly manifest. The name is the TRUE identity (bag keys, source
    configs, reports); type_id is a declared convention handle."""
    name: str
    type_id: int
    vocab_size: int      # LOCAL band size [0, size); global offset assigned by the layout
    tokenizer: Any = None  # the modality's LOCAL-ID producer (BPE codec / VQ codec / name table)


def build_layout(modalities: Sequence[Modality]) -> VocabLayout:
    """THE ASSEMBLER: stack bands in list order; offsets fall out of the stacking.

    Validates the declarations (the assembler never INVENTS ids — uniqueness
    checks only; each modality self-reports its canonical type_id)."""
    names = [m.name for m in modalities]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate modality names: {names}")
    reserved = set(names) & RESERVED_BAG_KEYS
    if reserved:
        raise ValueError(f"modality name(s) collide with reserved bag keys: {sorted(reserved)}")
    type_ids = [m.type_id for m in modalities]
    if len(set(type_ids)) != len(type_ids):
        raise ValueError(f"duplicate type_ids: {[(m.name, m.type_id) for m in modalities]}")

    layout = VocabLayout()
    offset = 0
    for m in modalities:
        if m.vocab_size <= 0:
            raise ValueError(f"modality '{m.name}' declares empty band (size {m.vocab_size})")
        layout.add_range(m.type_id, offset, offset + m.vocab_size)
        offset += m.vocab_size
    return layout
