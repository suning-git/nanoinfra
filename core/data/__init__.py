"""Nanoinfra data loading components (mechanisms only — each concrete source
lives with its modality and is injected via MixedDataLoader's source_types)."""

from .synthetic_loader import synthetic_token_loader
from .sequence_recipe import SequenceRecipe
from .dist_sampler import ResumableDistributedSampler
from .supervision import (
    SupervisionStrategy,
    NextTokenPrediction,
    AlignedSupervision,
)
from .data_source import DataSource
from .mixed_dataloader import MixedDataLoader

__all__ = [
    "synthetic_token_loader",
    "SequenceRecipe",
    "ResumableDistributedSampler",
    "SupervisionStrategy",
    "NextTokenPrediction",
    "AlignedSupervision",
    "DataSource",
    "MixedDataLoader",
]
