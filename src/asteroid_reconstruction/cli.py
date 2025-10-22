"""Command line entry point for asteroid reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .lightcurve import load_lightcurve_csv
from .optimizer import LightcurveInversion
from .shape import ConvexShapeModel


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a convex asteroid shape from lightcurve data."
    )
    parser.add_argument("input", type=Path, help="CSV file containing the lightcurve")
    parser.add_argument("output", type=Path, help="Destination OBJ mesh file")
    parser.add_argument(
        "--iterations", type=int, default=80, help="Number of optimisation iterations"
    )
    parser.add_argument(
        "--subdivisions", type=int, default=1, help="Icosahedron subdivision depth"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.1, help="Gradient descent step size"
    )
    parser.add_argument(
        "--regularisation",
        type=float,
        default=0.02,
        help="Smoothness weight applied to neighbouring radii",
    )
    parser.add_argument(
        "--lambert-weight",
        type=float,
        default=0.5,
        help="Blend between Lambertian (0) and Lommel-Seeliger (1) scattering",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Finite difference step for sensitivity estimation",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=0.05,
        help="Lower bound applied to facet radii after each update",
    )
    parser.add_argument(
        "--initial-radius",
        type=float,
        default=1.0,
        help="Initial radial scale of the convex model",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV file to store the optimisation loss history",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration logging and only print the final summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    observations = load_lightcurve_csv(args.input)
    if not observations:
        raise SystemExit("No observations found in input CSV")

    shape = ConvexShapeModel.from_subdivision(args.subdivisions, scale=args.initial_radius)
    inversion = LightcurveInversion(
        shape,
        observations,
        lambert_weight=args.lambert_weight,
        finite_difference_epsilon=args.epsilon,
    )

    def log_progress(record) -> None:
        if not args.quiet:
            print(
                f"Iter {record.iteration:03d}: loss={record.loss:.6f} "
                f"(data={record.data_loss:.6f}, reg={record.regularisation_loss:.6f})"
            )

    history = inversion.fit(
        iterations=args.iterations,
        step_size=args.step_size,
        regularisation=args.regularisation,
        history_callback=log_progress if not args.quiet else None,
        report_path=args.report,
        min_radius=args.min_radius,
    )

    shape.export_obj(args.output)
    final = history[-1]
    radii = shape.radii
    print(
        "Reconstruction complete:\n"
        f"  iterations: {len(history)}\n"
        f"  final loss: {final.loss:.6f}\n"
        f"  data loss: {final.data_loss:.6f}\n"
        f"  regularisation loss: {final.regularisation_loss:.6f}\n"
        f"  radius range: [{radii.min():.3f}, {radii.max():.3f}]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
