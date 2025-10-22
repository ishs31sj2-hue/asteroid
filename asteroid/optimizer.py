"""Kaasalainen-style lightcurve inversion optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .lightcurve import LightcurveData
from .mesh import Icosphere
from .photometry import ConvexShapeModel, SpinState, simulate_lightcurve


@dataclass
class InversionResult:
    radii: np.ndarray
    spin: SpinState
    cost_history: List[float]
    residual_history: List[float]


class KaasalainenInversion:
    """Perform convex inversion using a simple gradient-descent scheme."""

    def __init__(
        self,
        mesh_subdivisions: int = 1,
        smoothness: float = 1e-2,
        learning_rate: float = 0.05,
        spin_learning_rate: float = 5e-4,
        finite_difference_eps: float = 1e-3,
        max_iterations: int = 200,
        optimize_spin: bool = True,
        verbose: bool = False,
    ) -> None:
        self.mesh_subdivisions = mesh_subdivisions
        self.smoothness = smoothness
        self.learning_rate = learning_rate
        self.spin_learning_rate = spin_learning_rate
        self.finite_difference_eps = finite_difference_eps
        self.max_iterations = max_iterations
        self.optimize_spin = optimize_spin
        self.verbose = verbose

    def fit(
        self,
        lightcurve: LightcurveData,
        initial_spin: SpinState,
        initial_radii: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> InversionResult:
        mesh = Icosphere(subdivisions=self.mesh_subdivisions)
        model = ConvexShapeModel(mesh, radii=initial_radii)
        spin = initial_spin
        weights = lightcurve.residual_weights

        cost_history: List[float] = []
        rms_history: List[float] = []

        for iteration in range(self.max_iterations):
            model.ensure_positive()
            synthetic = simulate_lightcurve(
                model,
                spin,
                lightcurve.time,
                lightcurve.sun_vectors,
                lightcurve.observer_vectors,
            )
            residual = synthetic - lightcurve.brightness
            weighted_residual = residual * np.sqrt(weights)
            data_cost = 0.5 * np.sum(weighted_residual**2)
            reg_cost, reg_grad = model.regularization(self.smoothness)
            cost = data_cost + reg_cost
            rms = float(np.sqrt(np.mean(residual**2)))
            cost_history.append(float(cost))
            rms_history.append(rms)

            if self.verbose:
                print(
                    f"Iter {iteration:03d}: cost={cost:.6g} rms={rms:.4g}"
                )

            if callback is not None:
                callback(
                    iteration,
                    {
                        "cost": cost,
                        "data_cost": data_cost,
                        "reg_cost": reg_cost,
                        "rms": rms,
                    },
                )

            grad_radii = self._gradient_radii(
                model,
                spin,
                lightcurve,
                residual,
                weights,
                synthetic,
            )
            total_grad = grad_radii + reg_grad
            model.radii -= self.learning_rate * total_grad

            if self.optimize_spin:
                spin = self._update_spin(
                    model,
                    spin,
                    lightcurve,
                    residual,
                    weights,
                    synthetic,
                )

        return InversionResult(model.radii.copy(), spin, cost_history, rms_history)

    def _gradient_radii(
        self,
        model: ConvexShapeModel,
        spin: SpinState,
        lightcurve: LightcurveData,
        residual: np.ndarray,
        weights: np.ndarray,
        synthetic: np.ndarray,
    ) -> np.ndarray:
        grad = np.zeros_like(model.radii)
        base_brightness = synthetic
        eps = self.finite_difference_eps
        for idx in range(len(model.radii)):
            original = model.radii[idx]
            model.radii[idx] = original + eps
            model.ensure_positive()
            perturbed = simulate_lightcurve(
                model,
                spin,
                lightcurve.time,
                lightcurve.sun_vectors,
                lightcurve.observer_vectors,
            )
            derivative = (perturbed - base_brightness) / eps
            grad[idx] = np.sum(weights * residual * derivative)
            model.radii[idx] = original
        model.ensure_positive()
        return grad

    def _update_spin(
        self,
        model: ConvexShapeModel,
        spin: SpinState,
        lightcurve: LightcurveData,
        residual: np.ndarray,
        weights: np.ndarray,
        synthetic: np.ndarray,
    ) -> SpinState:
        eps = self.finite_difference_eps
        params = np.array(
            [spin.rotation_period, spin.pole_ra, spin.pole_dec, spin.initial_phase]
        )
        grads = np.zeros_like(params)
        for i in range(len(params)):
            shifted = params.copy()
            shifted[i] += eps
            trial_spin = SpinState(
                rotation_period=shifted[0],
                pole_ra=shifted[1],
                pole_dec=shifted[2],
                initial_phase=shifted[3],
                epoch=spin.epoch,
            )
            perturbed = simulate_lightcurve(
                model,
                trial_spin,
                lightcurve.time,
                lightcurve.sun_vectors,
                lightcurve.observer_vectors,
            )
            derivative = (perturbed - synthetic) / eps
            grads[i] = np.sum(weights * residual * derivative)
        updated = params - self.spin_learning_rate * grads
        updated_spin = SpinState(
            rotation_period=float(max(updated[0], eps)),
            pole_ra=float(updated[1]),
            pole_dec=float(np.clip(updated[2], -np.pi / 2 + 1e-4, np.pi / 2 - 1e-4)),
            initial_phase=float(updated[3]),
            epoch=spin.epoch,
        )
        return updated_spin
