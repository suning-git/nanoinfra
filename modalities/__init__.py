"""
modalities — the modality implementation namespace (one folder per modality).

Each `modalities/<name>/` exports `manifest(...) -> Modality` (its assembly
registration form; see assembler.py) plus whatever convenience constants and
implementation its own data sources need. Core NEVER imports this package;
orchestrators compose manifests into a VocabLayout + control resolver at
assembly time.
"""
