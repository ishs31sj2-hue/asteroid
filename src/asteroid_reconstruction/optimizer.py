"""Lightcurve inversion using a Kaasalainen-style algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

from .lightcurve import LightcurveObservation
from .shape import ConvexShapeModel


def scattering_response(
    normals: np.ndarray,
    areas: np.ndarray,
    sun_vector: np.ndarray,
    observer_vector: np.ndarray,
    lambert_weight: float,
) -> float:
    """Compute the disk-integrated brightness for a single geometry."""

    cos_sun = normals @ sun_vector
    cos_obs = normals @ observer_vector
    mask = (cos_sun > 0) & (cos_obs > 0)
    if not np.any(mask):
        return 0.0
    cos_sun = cos_sun[mask]
    cos_obs = cos_obs[mask]
    illum = (1.0 - lambert_weight) * cos_sun
    denom = np.maximum(cos_sun + cos_obs, 1e-8)
    ls_term = lambert_weight * (cos_sun * cos_obs / denom)
    contribution = areas[mask] * (illum + ls_term)
    return float(np.sum(contribution))


def render_lightcurve(
    shape: ConvexShapeModel,
    sun_vectors: np.ndarray,
    observer_vectors: np.ndarray,
    lambert_weight: float = 0.5,
) -> np.ndarray:
    normals, areas, _ = shape.facet_geometry()
    brightness = np.empty(len(sun_vectors), dtype=float)
    for idx, (sun_vec, obs_vec) in enumerate(zip(sun_vectors, observer_vectors, strict=True)):
        brightness[idx] = scattering_response(normals, areas, sun_vec, obs_vec, lambert_weight)
    return brightness


@dataclass
class OptimisationHistory:
    iteration: int
    loss: float
    data_loss: float
    regularisation_loss: float


class LightcurveInversion:
    """Kaasalainen-style convex lightcurve inversion."""

    def __init__(
        self,
        shape: ConvexShapeModel,
        observations: Sequence[LightcurveObservation],
        lambert_weight: float = 0.5,
        finite_difference_epsilon: float = 1e-3,
    ) -> None:
        if not observations:
            raise ValueError("At least one observation is required")
        self.shape = shape
        self.observations = list(observations)
        self.lambert_weight = float(lambert_weight)
        self.eps = float(finite_difference_epsilon)
        self._sun_vectors = np.stack([obs.sun_vector for obs in self.observations], axis=0)
        self._observer_vectors = np.stack(
            [obs.observer_vector for obs in self.observations], axis=0
        )
        self._brightness = np.array([obs.brightness for obs in self.observations], dtype=float)

    def _sensitivities(self) -> tuple[np.ndarray, np.ndarray]:
        base = render_lightcurve(
            self.shape, self._sun_vectors, self._observer_vectors, self.lambert_weight
        )
        n_params = len(self.shape.radii)
        sensitivities = np.empty((n_params, len(self.observations)), dtype=float)
        for idx in range(n_params):
            original = self.shape.radii[idx]
            self.shape.radii[idx] = original + self.eps
            self.shape.clamp_radii(minimum=1e-3)
            perturbed = render_lightcurve(
                self.shape, self._sun_vectors, self._observer_vectors, self.lambert_weight
            )
            sensitivities[idx] = (perturbed - base) / self.eps
            self.shape.radii[idx] = original
        return base, sensitivities

    def fit(
        self,
        iterations: int = 80,
        step_size: float = 0.1,
        regularisation: float = 0.02,
        history_callback: Callable[[OptimisationHistory], None] | None = None,
        report_path: str | Path | None = None,
        min_radius: float = 0.05,
    ) -> List[OptimisationHistory]:
        history: List[OptimisationHistory] = []
        report_file = Path(report_path) if report_path is not None else None
        if report_file is not None:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            fh = report_file.open("w", encoding="utf8")
        else:
            fh = None

        try:
            for iteration in range(1, iterations + 1):
                base_brightness, sensitivities = self._sensitivities()
                residuals = base_brightness - self._brightness
                data_loss = 0.5 * np.mean(residuals**2)
                reg_loss, reg_grad = self.shape.regularisation(regularisation)
                grad = (sensitivities @ residuals) / len(self.observations)
                grad += reg_grad
                self.shape.radii -= step_size * grad
                self.shape.clamp_radii(minimum=min_radius)
                total_loss = data_loss + reg_loss

                record = OptimisationHistory(
                    iteration=iteration,
                    loss=total_loss,
                    data_loss=data_loss,
                    regularisation_loss=reg_loss,
                )
                history.append(record)
                if history_callback is not None:
                    history_callback(record)
                if fh is not None:
                    fh.write(
                        f"{iteration},{total_loss:.6f},{data_loss:.6f},{reg_loss:.6f}\n"
                    )
            return history
        finally:
            if fh is not None:
                fh.close()
