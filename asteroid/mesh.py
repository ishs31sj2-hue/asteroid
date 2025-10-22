"""Mesh primitives for convex asteroid models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np


@dataclass
class Icosphere:
    """Simple triangular mesh based on a subdivided icosahedron."""

    subdivisions: int = 1

    def __post_init__(self) -> None:
        vertices, faces = self._generate_icosahedron()
        for _ in range(self.subdivisions):
            vertices, faces = self._subdivide(vertices, faces)
        self.vertices = self._normalize_vertices(vertices)
        self.faces = faces

    @staticmethod
    def _generate_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
        t = (1.0 + 5 ** 0.5) / 2.0
        vertices = np.array(
            [
                (-1, t, 0),
                (1, t, 0),
                (-1, -t, 0),
                (1, -t, 0),
                (0, -1, t),
                (0, 1, t),
                (0, -1, -t),
                (0, 1, -t),
                (t, 0, -1),
                (t, 0, 1),
                (-t, 0, -1),
                (-t, 0, 1),
            ],
            dtype=float,
        )
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
        return vertices, faces

    @classmethod
    def _subdivide(
        cls, vertices: np.ndarray, faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        new_faces = []
        vertices_list = vertices.tolist()

        def midpoint(i: int, j: int) -> int:
            key = tuple(sorted((i, j)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            vi = np.array(vertices_list[i])
            vj = np.array(vertices_list[j])
            vm = (vi + vj) / 2.0
            vertices_list.append(vm.tolist())
            idx = len(vertices_list) - 1
            midpoint_cache[key] = idx
            return idx

        for tri in faces:
            v1, v2, v3 = tri
            a = midpoint(v1, v2)
            b = midpoint(v2, v3)
            c = midpoint(v3, v1)
            new_faces.extend(
                [
                    (v1, a, c),
                    (v2, b, a),
                    (v3, c, b),
                    (a, b, c),
                ]
            )
        return np.array(vertices_list, dtype=float), np.array(new_faces, dtype=int)

    @staticmethod
    def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms

    @property
    def face_normals(self) -> np.ndarray:
        return compute_normals(self.vertices, self.faces)

    @property
    def face_areas(self) -> np.ndarray:
        return compute_face_areas(self.vertices, self.faces)

    def scaled_vertices(self, radii: np.ndarray) -> np.ndarray:
        radii = np.asarray(radii, dtype=float)
        if radii.shape != (len(self.vertices),):
            raise ValueError("radii must match the number of vertices")
        return self.vertices * radii[:, None]


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    normals = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


@lru_cache(maxsize=1024)
def vertex_adjacency(num_vertices: int, faces: Tuple[Tuple[int, int, int], ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Return CSR-like adjacency lists for the mesh."""

    neighbors = [[] for _ in range(num_vertices)]
    for f in faces:
        a, b, c = f
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    indptr = [0]
    indices = []
    for neigh in neighbors:
        uniq = sorted(set(neigh))
        indices.extend(uniq)
        indptr.append(len(indices))
    return np.array(indptr, dtype=int), np.array(indices, dtype=int)
