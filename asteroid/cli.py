"""Command-line interface for running lightcurve inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .lightcurve import LightcurveLoader
from .optimizer import KaasalainenInversion
from .photometry import SpinState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct a convex asteroid model from a lightcurve."
    )
    parser.add_argument("lightcurve", type=Path, help="CSV file containing the lightcurve")
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=1,
        help="Number of recursive subdivisions for the icosphere mesh",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Number of optimizer iterations",
    )
    parser.add_argument(
        "--rotation-period",
        type=float,
        default=4.0,
        help="Initial rotation period in the same time units as the lightcurve",
    )
    parser.add_argument(
        "--pole-ra",
        type=float,
        default=0.0,
        help="Initial spin-pole right ascension (radians)",
    )
    parser.add_argument(
        "--pole-dec",
        type=float,
        default=0.5,
        help="Initial spin-pole declination (radians)",
    )
    parser.add_argument(
        "--initial-phase",
        type=float,
        default=0.0,
        help="Initial rotation phase at the reference epoch (radians)",
    )
    parser.add_argument(
        "--epoch",
        type=float,
        default=0.0,
        help="Reference epoch of the initial phase",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=1e-2,
        help="Regularization strength for the Laplacian smoothness prior",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for the shape parameters",
    )
    parser.add_argument(
        "--spin-learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for the spin parameters",
    )
    parser.add_argument(
        "--finite-difference-eps",
        type=float,
        default=1e-3,
        help="Finite difference step for gradient estimation",
    )
    parser.add_argument(
        "--no-spin-optimization",
        action="store_true",
        help="Disable refinement of the spin parameters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file where the recovered model parameters are stored",
    )
    return parser


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    loader = LightcurveLoader()
    data = loader.from_csv(args.lightcurve)
    spin = SpinState(
        rotation_period=args.rotation_period,
        pole_ra=args.pole_ra,
        pole_dec=args.pole_dec,
        initial_phase=args.initial_phase,
        epoch=args.epoch,
    )
    optimizer = KaasalainenInversion(
        mesh_subdivisions=args.subdivisions,
        smoothness=args.smoothness,
        learning_rate=args.learning_rate,
        spin_learning_rate=args.spin_learning_rate,
        finite_difference_eps=args.finite_difference_eps,
        max_iterations=args.iterations,
        optimize_spin=not args.no_spin_optimization,
        verbose=True,
    )
    result = optimizer.fit(data, spin)
    output = {
        "radii": result.radii.tolist(),
        "rotation_period": result.spin.rotation_period,
        "pole_ra": result.spin.pole_ra,
        "pole_dec": result.spin.pole_dec,
        "initial_phase": result.spin.initial_phase,
        "cost_history": result.cost_history,
        "residual_history": result.residual_history,
    }
    if args.output is not None:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    return output


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    main()
