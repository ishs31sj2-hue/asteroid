"""Utilities for working with lightcurve data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class LightcurveObservation:
    """Single photometric measurement and associated geometry."""

    time: float
    brightness: float
    sun_vector: np.ndarray
    observer_vector: np.ndarray

    def __post_init__(self) -> None:
        self.sun_vector = np.asarray(self.sun_vector, dtype=float)
        self.observer_vector = np.asarray(self.observer_vector, dtype=float)
        if self.sun_vector.shape != (3,) or self.observer_vector.shape != (3,):
            raise ValueError("Sun and observer vectors must be 3D")


def load_lightcurve_csv(path: str | Path) -> List[LightcurveObservation]:
    """Load observations from a CSV file.

    Parameters
    ----------
    path:
        CSV file path with the columns
        ``time,brightness,sun_x,sun_y,sun_z,observer_x,observer_y,observer_z``.
    """

    path = Path(path)
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
    )
    if data.ndim == 0:
        data = np.array([data])

    observations: List[LightcurveObservation] = []
    for row in data:
        sun_vec = np.array([row["sun_x"], row["sun_y"], row["sun_z"]], dtype=float)
        obs_vec = np.array(
            [row["observer_x"], row["observer_y"], row["observer_z"]], dtype=float
        )
        observations.append(
            LightcurveObservation(
                time=float(row["time"]),
                brightness=float(row["brightness"]),
                sun_vector=sun_vec,
                observer_vector=obs_vec,
            )
        )
    return observations


def observations_to_arrays(observations: Iterable[LightcurveObservation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of observations into structured arrays."""

    times = np.array([obs.time for obs in observations], dtype=float)
    brightness = np.array([obs.brightness for obs in observations], dtype=float)
    sun_vectors = np.stack([obs.sun_vector for obs in observations], axis=0)
    observer_vectors = np.stack([obs.observer_vector for obs in observations], axis=0)
    return times, brightness, np.stack([sun_vectors, observer_vectors], axis=1)
