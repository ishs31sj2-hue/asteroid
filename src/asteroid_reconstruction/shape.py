"""Geometry helpers and the convex shape model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


PHI = (1 + 5 ** 0.5) / 2


def create_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangular faces of a unit icosahedron."""

    verts = np.array(
        [
            (-1, PHI, 0),
            (1, PHI, 0),
            (-1, -PHI, 0),
            (1, -PHI, 0),
            (0, -1, PHI),
            (0, 1, PHI),
            (0, -1, -PHI),
            (0, 1, -PHI),
            (PHI, 0, -1),
            (PHI, 0, 1),
            (-PHI, 0, -1),
            (-PHI, 0, 1),
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=int,
    )
    return verts, faces


def midpoint_unit(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    mid = (v1 + v2) * 0.5
    return mid / np.linalg.norm(mid)


def subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Loop subdivision of a triangular mesh on the sphere."""

    midpoint_cache: dict[tuple[int, int], int] = {}
    vertices_list = vertices.tolist()
    new_faces: list[tuple[int, int, int]] = []

    def midpoint_index(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        vi = np.array(vertices_list[i])
        vj = np.array(vertices_list[j])
        mid = midpoint_unit(vi, vj)
        vertices_list.append(mid.tolist())
        idx = len(vertices_list) - 1
        midpoint_cache[key] = idx
        return idx

    for tri in faces:
        i, j, k = tri
        a = midpoint_index(i, j)
        b = midpoint_index(j, k)
        c = midpoint_index(k, i)
        new_faces.extend(
            [
                (i, a, c),
                (a, j, b),
                (c, b, k),
                (a, b, c),
            ]
        )

    new_vertices = np.array(vertices_list, dtype=float)
    new_vertices /= np.linalg.norm(new_vertices, axis=1, keepdims=True)
    return new_vertices, np.array(new_faces, dtype=int)


def create_icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = create_icosahedron()
    for _ in range(subdivisions):
        verts, faces = subdivide(verts, faces)
    return verts, faces


def extract_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in faces:
        i, j, k = tri
        pairs = [(i, j), (j, k), (k, i)]
        for a, b in pairs:
            edge = (a, b) if a < b else (b, a)
            edges.add(edge)
    return np.array(sorted(edges), dtype=int)


@dataclass
class ConvexShapeModel:
    directions: np.ndarray
    faces: np.ndarray
    radii: np.ndarray
    edges: np.ndarray

    @classmethod
    def from_subdivision(cls, subdivisions: int, scale: float = 1.0) -> "ConvexShapeModel":
        directions, faces = create_icosphere(subdivisions)
        radii = np.full(len(directions), float(scale))
        edges = extract_edges(faces)
        return cls(directions=directions, faces=faces, radii=radii, edges=edges)

    @property
    def vertices(self) -> np.ndarray:
        return self.directions * self.radii[:, None]

    def facet_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        verts = self.vertices
        tri = self.faces
        v1 = verts[tri[:, 0]]
        v2 = verts[tri[:, 1]]
        v3 = verts[tri[:, 2]]
        normals = np.cross(v2 - v1, v3 - v1)
        areas = 0.5 * np.linalg.norm(normals, axis=1)
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        normals_unit = normals / normals_norm
        centers = (v1 + v2 + v3) / 3.0
        return normals_unit, areas, centers

    def regularisation(self, weight: float) -> tuple[float, np.ndarray]:
        if weight <= 0:
            return 0.0, np.zeros_like(self.radii)
        diffs = self.radii[self.edges[:, 0]] - self.radii[self.edges[:, 1]]
        loss = 0.5 * weight * np.mean(diffs**2)
        grad = np.zeros_like(self.radii)
        if len(self.edges) == 0:
            return loss, grad
        scale = weight / len(self.edges)
        for (i, j), diff in zip(self.edges, diffs, strict=False):
            grad[i] += scale * diff
            grad[j] -= scale * diff
        return loss, grad

    def clamp_radii(self, minimum: float = 1e-3, maximum: float | None = None) -> None:
        if maximum is None:
            np.maximum(self.radii, minimum, out=self.radii)
        else:
            np.clip(self.radii, minimum, maximum, out=self.radii)

    def export_obj(self, path: str | Path) -> None:
        path = Path(path)
        verts = self.vertices
        with path.open("w", encoding="utf8") as fh:
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.faces:
                fh.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
