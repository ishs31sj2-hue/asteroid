"""Generate a synthetic lightcurve for testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from asteroid_reconstruction.optimizer import render_lightcurve
from asteroid_reconstruction.shape import ConvexShapeModel


def random_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2 * np.pi, size=count)
    cos_theta = rng.uniform(-1.0, 1.0, size=count)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    return np.stack([x, y, z], axis=1)


def create_ellipsoid(subdivisions: int = 1) -> ConvexShapeModel:
    shape = ConvexShapeModel.from_subdivision(subdivisions, scale=1.0)
    axes = np.array([1.0, 0.8, 0.6])
    for idx, direction in enumerate(shape.directions):
        denom = np.sqrt(
            (direction[0] / axes[0]) ** 2
            + (direction[1] / axes[1]) ** 2
            + (direction[2] / axes[2]) ** 2
        )
        shape.radii[idx] = 1.0 / denom
    shape.clamp_radii(minimum=0.1)
    return shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic lightcurve")
    parser.add_argument("--output", type=Path, default=Path("sample_lightcurve.csv"))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--noise", type=float, default=0.02, help="Relative noise level")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = create_ellipsoid(subdivisions=1)
    sun_vectors = random_unit_vectors(args.samples, rng)
    observer_vectors = random_unit_vectors(args.samples, rng)
    brightness = render_lightcurve(shape, sun_vectors, observer_vectors, lambert_weight=0.6)
    noise = rng.normal(scale=args.noise * np.mean(brightness), size=args.samples)
    brightness += noise

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8") as fh:
        fh.write("time,brightness,sun_x,sun_y,sun_z,observer_x,observer_y,observer_z\n")
        for idx in range(args.samples):
            fh.write(
                f"{idx:.3f},{brightness[idx]:.6f},{sun_vectors[idx,0]:.6f},{sun_vectors[idx,1]:.6f},{sun_vectors[idx,2]:.6f},"
                f"{observer_vectors[idx,0]:.6f},{observer_vectors[idx,1]:.6f},{observer_vectors[idx,2]:.6f}\n"
            )

    print(f"Wrote {args.samples} samples to {output}")


if __name__ == "__main__":
    main()
