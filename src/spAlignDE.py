"""Case-preserving public interface for spAlignDE."""

from __future__ import annotations

import importlib
import sys

import spalignde as _implementation

__all__ = _implementation.__all__
__path__: list[str] = []


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


# Preserve imports for users who access public subpackages directly.
for _submodule in ("alignment", "clustering", "datasets", "inference", "io", "uncertainty"):
    _module = importlib.import_module(f"spalignde.{_submodule}")
    globals()[_submodule] = _module
    sys.modules[f"{__name__}.{_submodule}"] = _module
