"""
motion.data — the reusable data leg (R6, 2026-07-07).

SMPL/BVH <-> rot139 converters, differentiable FK, dataset loaders, windowing,
and the core DataSources (MotionDataSource / T2MDataSource). Importing this
package runs the layout bootstrap (paths), so the data modules keep flat imports.

Import submodules explicitly (kept out of package init so `import modalities.motion`
stays light — the assembler path needs no torch/SMPL):
    from modalities.motion.data import dataset, paths
    from modalities.motion.data.sources import MotionDataSource, T2MDataSource
    from modalities.motion.data.converters import smpl_body, smpl_to_rot139
    from modalities.motion.data import fk_torch
"""

from modalities.motion.data import paths  # noqa: F401  (runs the sys.path bootstrap)
