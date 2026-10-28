"""Model comparison workflows for Balloon-Windkessel, PINN, and CoSTA."""

from importlib import import_module
from typing import Any

__all__ = ["default_output_dir", "load_csv", "run_comparison", "synthetic_data"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    target = "run" if name == "run_comparison" else name
    return getattr(import_module(".run", __name__), target)
