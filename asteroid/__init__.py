"""Tools for reconstructing convex asteroid models from lightcurve data.

This package provides a lightcurve inversion pipeline inspired by the
Kaasalainen algorithm, including utilities for loading lightcurve data,
constructing convex polyhedral shape models, simulating brightness curves,
and fitting model parameters to observations.
"""

from .lightcurve import LightcurveData, LightcurveLoader
from .mesh import Icosphere
from .optimizer import KaasalainenInversion
from .photometry import ConvexShapeModel, simulate_lightcurve

__all__ = [
    "LightcurveData",
    "LightcurveLoader",
    "Icosphere",
    "ConvexShapeModel",
    "simulate_lightcurve",
    "KaasalainenInversion",
]
