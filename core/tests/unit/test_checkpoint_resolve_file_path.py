"""An explicit checkpoint path that is a file is refused where it is resolved.

Only DCP checkpoint directories are loadable. A file used to be handed back unchanged, and the run then died
later, deep in loading, on <file>/meta.json.
"""

import pytest

from core.model import checkpoint_manager as cm


def test_a_file_path_is_refused(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"old single-file checkpoint")
    with pytest.raises(NotADirectoryError, match="single-file checkpoints are not supported"):
        cm.resolve_checkpoint_path(str(f))
