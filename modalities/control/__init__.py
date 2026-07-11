"""
control — the protocol modality.

Control tokens are content in NO modality: they are the sequence protocol —
the framing recipes use to structure sequences ACROSS modalities. A control
token's identity IS its canonical NAME (order = local id); `<|name|>` is just
the display/artifact rendering of that name (see display_form).

As a band owner control is a perfectly regular manifest entry; what is
system-level is its PRODUCT: the control resolver (name -> global id), built
at assembly time from control's name table x the layout's offset, riding the
shared bag as `control_resolver`. It is THE authority on protocol
names — contract: resolve(name) -> int | None (None = not a control token).
"""

from modalities.assembler import Modality

TYPE_ID = 2  # canonical (0=text, 1=motion, 2=control)

# The protocol registry. ORDER IS IDENTITY: local id = list position (the
# trained artifact reserves the trailing band in exactly this order).
CONTROL_TOKENS = [
    # Basic sequence delimiters
    "bos",
    "eos",
    # Region markers (text / motion boundaries)
    "text_start",
    "text_end",
    "motion_start",
    "motion_end",
    # Reserved for future experiments
    "ctrl0",
    "ctrl1",
    "ctrl2",
    "ctrl3",
    # Conversational format (finetuning)
    "user_start",
    "user_end",
    "assistant_start",
    "assistant_end",
    "python_start",
    "python_end",
    "output_start",
    "output_end",
]


def display_form(name: str) -> str:
    """Canonical name -> artifact/display rendering ("bos" -> "<|bos|>").
    The delimiters are control's convention; used at the trained-artifact
    boundary (BPE special strings) and in human-facing rendering."""
    return f"<|{name}|>"


class ControlTable:
    """control's LOCAL-ID producer: canonical name -> local id [0, n).
    Plays for control exactly the role the BPE codec plays for text."""

    def __init__(self, names):
        self.names = list(names)
        self._ids = {n: i for i, n in enumerate(self.names)}
        if len(self._ids) != len(self.names):
            raise ValueError("duplicate control token names")

    @property
    def vocab_size(self) -> int:
        return len(self.names)

    def encode(self, name: str) -> int:
        return self._ids[name]           # KeyError = unknown protocol name

    def get(self, name: str):
        return self._ids.get(name)       # None = not a control token

    def decode(self, local_id: int) -> str:
        return self.names[local_id]


def manifest() -> Modality:
    return Modality(name="control", type_id=TYPE_ID,
                    vocab_size=len(CONTROL_TOKENS),
                    tokenizer=ControlTable(CONTROL_TOKENS))


class ControlResolver:
    """The system-level protocol authority: canonical name -> GLOBAL id.
    Manufactured per assembly (control's table x the layout's offset) — not a
    singleton, not a registry; it rides the bag as `control_resolver`."""

    def __init__(self, table: ControlTable, offset: int):
        self._table = table
        self._offset = offset

    def resolve(self, name: str):
        """int global id, or None if `name` is not a control token."""
        local = self._table.get(name)
        return None if local is None else self._offset + local


def make_control_resolver(control: Modality, layout) -> ControlResolver:
    """control's post-assembly export: control brings the names, the layout
    brings the addresses; the orchestrator marries them."""
    return ControlResolver(control.tokenizer, layout.offset(control.type_id))
