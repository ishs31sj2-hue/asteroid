"""Photometric simulation tools for convex asteroid models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .mesh import Icosphere, compute_face_areas, compute_normals, vertex_adjacency


@dataclass
class SpinState:
    """Describe the spin orientation and period of the asteroid."""

    rotation_period: float
    pole_ra: float
    pole_dec: float
    initial_phase: float = 0.0
    epoch: float = 0.0

    def phase(self, time: np.ndarray) -> np.ndarray:
        return self.initial_phase + 2.0 * np.pi * (time - self.epoch) / self.rotation_period


class ConvexShapeModel:
    """Convex polyhedral asteroid model parameterized by vertex radii."""

    def __init__(
        self,
        mesh: Icosphere,
        radii: Iterable[float] | None = None,
        albedo: float = 1.0,
    ) -> None:
        self.mesh = mesh
        if radii is None:
            radii = np.ones(len(mesh.vertices))
        self.radii = np.asarray(radii, dtype=float)
        if self.radii.shape != (len(self.mesh.vertices),):
            raise ValueError("radii must match the number of mesh vertices")
        self.albedo = float(albedo)

    @property
    def vertex_positions(self) -> np.ndarray:
        return self.mesh.scaled_vertices(self.radii)

    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    def geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = self.vertex_positions
        normals = compute_normals(vertices, self.faces)
        areas = compute_face_areas(vertices, self.faces)
        return normals, areas

    def ensure_positive(self, min_radius: float = 1e-3) -> None:
        self.radii = np.clip(self.radii, min_radius, None)

    def regularization(self, strength: float = 1.0) -> Tuple[float, np.ndarray]:
        """Return (penalty, gradient) for Laplacian smoothness."""

        faces_tuple = tuple(map(tuple, self.faces.tolist()))
        indptr, indices = vertex_adjacency(len(self.radii), faces_tuple)
        penalty = 0.0
        grad = np.zeros_like(self.radii)
        for i in range(len(self.radii)):
            start, end = indptr[i], indptr[i + 1]
            neigh = indices[start:end]
            if len(neigh) == 0:
                continue
            diff = self.radii[i] - self.radii[neigh]
            penalty += 0.5 * np.sum(diff**2)
            grad[i] += np.sum(diff)
            grad[neigh] -= diff
        return strength * penalty, strength * grad

    def brightness_for_directions(
        self, sun_directions: np.ndarray, observer_directions: np.ndarray
    ) -> np.ndarray:
        normals, areas = self.geometry()
        sun_dot = sun_directions @ normals.T
        obs_dot = observer_directions @ normals.T
        illum = np.clip(sun_dot, 0.0, None) * np.clip(obs_dot, 0.0, None)
        return self.albedo * illum @ areas


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_align_z_to_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(vector, z):
        return np.eye(3)
    if np.allclose(vector, -z):
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, vector)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(z, vector), -1.0, 1.0))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def inertial_to_body_rotations(spin: SpinState, time: np.ndarray) -> np.ndarray:
    pole = np.array(
        [
            np.cos(spin.pole_dec) * np.cos(spin.pole_ra),
            np.cos(spin.pole_dec) * np.sin(spin.pole_ra),
            np.sin(spin.pole_dec),
        ]
    )
    R_align = rotation_align_z_to_vector(pole)
    phases = spin.phase(time)
    rotations = np.empty((len(time), 3, 3))
    for i, phi in enumerate(phases):
        R = rotation_matrix_z(phi)
        rotations[i] = R_align @ R
    return rotations


def simulate_lightcurve(
    model: ConvexShapeModel,
    spin: SpinState,
    time: np.ndarray,
    sun_vectors: np.ndarray,
    observer_vectors: np.ndarray,
) -> np.ndarray:
    rotations = inertial_to_body_rotations(spin, time)
    sun_body = np.einsum("tij,tj->ti", rotations.transpose((0, 2, 1)), sun_vectors)
    obs_body = np.einsum("tij,tj->ti", rotations.transpose((0, 2, 1)), observer_vectors)
    return model.brightness_for_directions(sun_body, obs_body)
