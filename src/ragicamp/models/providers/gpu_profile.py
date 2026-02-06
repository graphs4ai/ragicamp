"""GPU profile detection for optimized model loading.

Replaces hardcoded magic numbers with a structured dataclass
that auto-detects GPU capabilities.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUProfile:
    """GPU capability profile for optimizing vLLM settings.

    Use ``GPUProfile.detect()`` to auto-detect the current GPU tier
    and get recommended vLLM parameters.
    """

    tier: str  # "high", "mid", "low", "cpu"
    gpu_mem_gb: float
    max_num_seqs: Optional[int]
    max_num_batched_tokens: Optional[int]

    # Tier thresholds (GB)
    _HIGH_THRESHOLD: float = 160  # B200, A100-80GB×2, etc.
    _MID_THRESHOLD: float = 80  # A100-80GB, H100, etc.

    @classmethod
    def detect(cls) -> "GPUProfile":
        """Detect GPU and return an appropriate profile.

        Returns:
            GPUProfile with recommended settings for the detected GPU.
        """
        gpu_mem_gb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        if gpu_mem_gb >= cls._HIGH_THRESHOLD:
            return cls(
                tier="high",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=512,
                max_num_batched_tokens=32768,
            )
        elif gpu_mem_gb >= cls._MID_THRESHOLD:
            return cls(
                tier="mid",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=256,
                max_num_batched_tokens=16384,
            )
        elif gpu_mem_gb > 0:
            return cls(
                tier="low",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=None,  # Use vLLM defaults
                max_num_batched_tokens=None,
            )
        else:
            return cls(
                tier="cpu",
                gpu_mem_gb=0,
                max_num_seqs=None,
                max_num_batched_tokens=None,
            )
