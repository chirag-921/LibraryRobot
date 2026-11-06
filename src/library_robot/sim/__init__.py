"""Simulation helpers.

Imports are lazy so XML/configuration tools can be used without importing
MuJoCo at package-import time. The existing ``from library_robot.sim import
load_model`` API is preserved.
"""


def load_model(*args, **kwargs):
    from .load_model import load_model as _load_model
    return _load_model(*args, **kwargs)


__all__ = ["load_model"]
