"""Random-state controls for reproducible spAlignDE workflows."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_random_seed(
    random_state: int,
    *,
    deterministic_torch: bool = False,
    warn_only: bool = True,
) -> dict[str, Any]:
    """Reset Python, NumPy, and PyTorch random generators.

    This function controls random generators used after it is called. Python's
    hash seed is fixed only when ``PYTHONHASHSEED`` is set before Python starts;
    the returned metadata records its launch-time value instead of implying
    that it can be changed safely at runtime.

    Parameters
    ----------
    random_state
        Non-negative seed shared by Python, NumPy, and PyTorch.
    deterministic_torch
        Request deterministic PyTorch algorithms and deterministic cuDNN
        behavior. Some CUDA operations used by S-LDDMM have no deterministic
        implementation; with ``warn_only=True`` PyTorch warns and continues.
    warn_only
        Forwarded to ``torch.use_deterministic_algorithms``.
    """
    seed = int(random_state)
    if seed < 0:
        raise ValueError("random_state must be non-negative")

    random.seed(seed)
    np.random.seed(seed)

    torch_available = False
    torch_deterministic = False
    try:
        import torch

        torch_available = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch_deterministic = torch.are_deterministic_algorithms_enabled()
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass

    return {
        "random_state": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_available": torch_available,
        "torch_seeded": torch_available,
        "torch_deterministic_algorithms": torch_deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


__all__ = ["set_random_seed"]
