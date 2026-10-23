"""The data snapshot's format change of 2026-10.

What changed. A checkpoint's meta.json keeps the loader's data snapshot under "dataloader_state". Until this change
it was a list of source states matched to the loader's sources by position:

    {"source_states": [state_0, state_1, ...], "accum": ...}   # nanoinfra 0.1.0 and 0.2.0 (the old loader)
    {"source_states": [state_0, state_1, ...]}                  # development versions between 0.2.0 and this change

Since the change it is a dict keyed by source name, {"source_states": {name: state}}, and the
loader accepts only that: a list cannot show which state belongs to which source, and two sources of the same type
that trade places would silently get each other's position.

What this file does. It converts a list snapshot into the name-keyed one, with the order given by the caller —
which source each list entry belongs to is known only from the old run (the order of `data.sources` in its config
for a from_config entry, the order of the list in its code otherwise); nothing here can check it. "accum" (the old
loader's mixing phase) is dropped: mixing counts are never restored.

    from core import compatibility
    loader = compatibility.accept_list_snapshot(loader)                    # one source: no order needed
    compatibility.upgrade_list_snapshot(ckpt_dir, ["text", "t2m"])         # rewrite meta.json once, with a backup
    # or: python -m core.compatibility.loader_snapshot <checkpoint_dir> text t2m

accept_list_snapshot leaves the checkpoint as it is, so the old code can still read it; upgrade_list_snapshot
rewrites meta.json (weights untouched) after copying it to meta.json.bak. The order is: build the loader, wrap it,
then restore (the Trainer calls set_state, or the caller does).

upgrade_list_snapshot finds the list snapshot only at the top of "dataloader_state". A project that wraps the loader
in its own object, whose snapshot then holds the loader's inside its own shape ({"loader": {...}, "stats": ...},
say), calls accept_list_snapshot on the inner MixedDataLoader instead; whatever the outer shape, the conversion
happens when the wrapper's set_state hands the inner snapshot to the loader's.

Delete this file when no checkpoint written before this change needs to be resumed.
"""
import argparse
import json
import os
import shutil

from core.utils import print0


def _order(source_order):
    if isinstance(source_order, str):
        raise TypeError(f"source_order is a list of source names, e.g. [{source_order!r}], not a string")
    order = list(source_order)
    if any(not isinstance(n, str) or not n for n in order):
        raise TypeError(f"source names are nonempty strings (they become the snapshot's JSON keys): {order}")
    repeated = sorted({n for n in order if order.count(n) > 1})
    if repeated:
        raise ValueError(f"source_order repeats {repeated}")
    return order


def _list_to_names(state, order):
    """{"source_states": [...] (, "accum": ...)} -> {"source_states": {name: state}} in the given order."""
    states = state["source_states"]
    if len(states) != len(order):
        raise ValueError(f"the old snapshot has {len(states)} source states but source_order names {len(order)} "
                         f"sources: {order}")
    pairs = ", ".join(f"{i} -> {name!r}" for i, name in enumerate(order))
    dropped = "; its 'accum' dropped" if "accum" in state else ""
    print0(f"  old list snapshot read by the given order: {pairs}{dropped}")
    return {"source_states": dict(zip(order, states))}


def accept_list_snapshot(loader, source_order=None):
    """Let `loader` restore from a list snapshot as well; returns the same loader.

    Its set_state is wrapped: a list snapshot is converted by `source_order` (the loader's source names, in the old
    list's order; may be left out when the loader has one source) and then restored as usual; a name-keyed snapshot
    is restored as usual. Call it after building the loader and before the Trainer restores the checkpoint. Nothing
    is written."""
    names = list(loader.sources)
    if source_order is None:
        if len(names) != 1:
            raise ValueError(f"source_order is needed: this loader has {len(names)} sources {names}, and only the "
                             f"old run tells which state in the list belongs to which")
        source_order = names
    order = _order(source_order)
    unknown = [n for n in order if n not in loader.sources]
    if unknown:
        raise ValueError(f"source_order names sources this loader does not have: {unknown} (it has {names})")
    restore = loader.set_state

    def set_state(state):
        if isinstance(state, dict) and isinstance(state.get("source_states"), list):
            state = _list_to_names(state, order)
        restore(state)

    loader.set_state = set_state
    return loader


def upgrade_list_snapshot(checkpoint_dir, source_order):
    """Rewrite checkpoint_dir/meta.json's list snapshot into the name-keyed one, after copying meta.json to
    meta.json.bak (an existing meta.json.bak is never overwritten: FileExistsError). Does nothing when meta.json
    holds no list snapshot. The weights are not touched; afterwards the old code cannot resume from this checkpoint
    unless meta.json.bak is put back."""
    path = os.path.join(checkpoint_dir, "meta.json")
    with open(path) as f:
        meta = json.load(f)
    snapshot = meta.get("dataloader_state")
    if not (isinstance(snapshot, dict) and isinstance(snapshot.get("source_states"), list)):
        print(f"{path}: no list-format data snapshot, nothing to do")
        return
    new = _list_to_names(snapshot, _order(source_order))
    backup = path + ".bak"
    if os.path.exists(backup):
        raise FileExistsError(f"{backup} already exists; move it away first (a backup is never overwritten)")
    shutil.copy2(path, backup)
    meta["dataloader_state"] = new
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)          # the checkpoint manager writes meta.json the same way
        f.flush()
        os.fsync(f.fileno())
    shutil.copymode(backup, tmp)
    os.replace(tmp, path)
    print(f"{path}: data snapshot rewritten by name ({list(new['source_states'])}); the original is {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rewrite a checkpoint's list-format data snapshot (written before the 2026-10 change) into "
                    "the name-keyed format; meta.json is copied to meta.json.bak first.")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("source_order", nargs="+", help="the loader's source names, in the old list's order")
    args = parser.parse_args(argv)
    upgrade_list_snapshot(args.checkpoint_dir, args.source_order)


if __name__ == "__main__":
    main()
