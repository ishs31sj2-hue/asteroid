"""Utilities for handling photometric lightcurve data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class LightcurveData:
    """Container for time-resolved lightcurve data.

    Attributes
    ----------
    time : np.ndarray
        Observation times expressed in the same units as the rotation period
        used in the inversion (typically hours).
    sun_vectors : np.ndarray
        Array with shape ``(N, 3)`` giving the direction from the asteroid to
        the Sun at each observation time in an inertial frame.
    observer_vectors : np.ndarray
        Array with shape ``(N, 3)`` giving the direction from the asteroid to
        the observer.
    magnitude : np.ndarray
        Observed brightness values. They may be magnitudes or already
        expressed as fluxes; the conversion is controlled by ``is_magnitude``.
    is_magnitude : bool
        True when the ``magnitude`` values are in astronomical magnitudes. In
        this case the :meth:`brightness` method converts them to linear flux
        units before comparison with the model.
    weights : Optional[np.ndarray]
        Optional weighting factors for each data point. When ``None`` each
        sample contributes equally to the cost function.
    """

    time: np.ndarray
    sun_vectors: np.ndarray
    observer_vectors: np.ndarray
    magnitude: np.ndarray
    is_magnitude: bool = True
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.sun_vectors = self._normalize(np.asarray(self.sun_vectors, dtype=float))
        self.observer_vectors = self._normalize(
            np.asarray(self.observer_vectors, dtype=float)
        )
        self.magnitude = np.asarray(self.magnitude, dtype=float)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != self.time.shape:
                raise ValueError("weights must match the shape of the time array")
        if not (self.time.shape == self.magnitude.shape == (len(self.sun_vectors),)):
            raise ValueError("Input arrays must share the same length")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Direction vectors must be non-zero")
        return vectors / norms

    @property
    def brightness(self) -> np.ndarray:
        """Return the lightcurve as linear flux values."""

        if self.is_magnitude:
            return 10 ** (-0.4 * self.magnitude)
        return self.magnitude

    @property
    def residual_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.ones_like(self.time)
        return self.weights


class LightcurveLoader:
    """Helper for loading lightcurve data from common file formats."""

    def __init__(self, delimiter: str = ",", comment: str = "#") -> None:
        self.delimiter = delimiter
        self.comment = comment

    def from_csv(
        self,
        path: Path | str,
        time_col: int = 0,
        sun_cols: Iterable[int] = (1, 2, 3),
        observer_cols: Iterable[int] = (4, 5, 6),
        mag_col: int = 7,
        weight_col: Optional[int] = None,
        is_magnitude: bool = True,
    ) -> LightcurveData:
        """Load a CSV file containing a lightcurve.

        The default column layout is compatible with synthetic data produced by
        the CLI included in this package. Users can adjust the column indices to
        match their own datasets.
        """

        path = Path(path)
        data = np.loadtxt(path, delimiter=self.delimiter, comments=self.comment)
        time = data[:, time_col]
        sun = data[:, list(sun_cols)]
        observer = data[:, list(observer_cols)]
        mag = data[:, mag_col]
        weights = data[:, weight_col] if weight_col is not None else None
        return LightcurveData(
            time=time,
            sun_vectors=sun,
            observer_vectors=observer,
            magnitude=mag,
            is_magnitude=is_magnitude,
            weights=weights,
        )
