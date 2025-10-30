"""
NAS-HGM: Neural Architecture Search via Huxley-Gödel Machine
Unbounded self-improving architecture search with automatic primitive synthesis.
"""

__version__ = "3.0.0"

from .architectures import (
    create_transformer_baseline,
    create_simple_mlp,
    create_bottleneck_arch,
    ArchitectureWrapper,
    mutate_architecture,
)

from .compression_eval import (
    evaluate_architecture_fast,
    compare_architectures,
)

from .arch_search import ArchitectureHGM

__all__ = [
    "create_transformer_baseline",
    "create_simple_mlp",
    "create_bottleneck_arch",
    "ArchitectureWrapper",
    "mutate_architecture",
    "evaluate_architecture_fast",
    "compare_architectures",
    "ArchitectureHGM",
]
