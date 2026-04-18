"""EdgeMoE — Expert-aware MoE inference engine for consumer hardware."""

__version__ = "0.1.0"

from edgemoe.hf_engine import HFEngine, HFEngineConfig
from edgemoe.engine import EdgeMoE as CustomEngine

# HFEngine is the default / recommended path (wraps HuggingFace transformers).
# CustomEngine is the research "from-scratch transformer" path — incomplete,
# kept for future work on custom attention kernels + TurboQuant integration.
EdgeMoE = HFEngine

__all__ = ["EdgeMoE", "HFEngine", "HFEngineConfig", "CustomEngine", "__version__"]
