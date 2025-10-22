# Kaasalainen Lightcurve Inversion Toolkit

This repository provides a lightweight, self-contained implementation of a
Kaasalainen-style convex inversion algorithm for reconstructing the shape of an
asteroid from time-resolved photometric observations. The code builds a
triangular mesh model of the body, simulates lightcurves assuming Lambertian
scattering, and iteratively adjusts the surface radii (and optionally the spin
state) to minimize the residual with respect to an observed lightcurve.

## Features

- Icosphere-based convex polyhedron representation with vertex-scaled radii.
- Lambertian lightcurve synthesis driven by Sun and observer geometry.
- Finite-difference gradient descent optimizer with Laplacian smoothness prior.
- Command-line interface for batch reconstruction and JSON export of results.

## Installation

The project uses standard library modules plus `numpy`. Install dependencies and
run the CLI via:

```bash
pip install -r requirements.txt  # if you add extra dependencies
python -m asteroid.cli path/to/lightcurve.csv --iterations 150
```

## Lightcurve Format

The loader expects a CSV file with the following columns (the delimiter and
layout can be customized through CLI flags):

1. Observation time (same units as the rotation period).
2. Sun direction vector components (x, y, z) in an inertial frame.
3. Observer direction vector components (x, y, z).
4. Observed magnitude (or flux values when `--magnitudes false` is implemented).

## Usage Example

```bash
python -m asteroid.cli synthetic_lightcurve.csv \
    --rotation-period 4.25 \
    --pole-ra 0.9 \
    --pole-dec 0.6 \
    --iterations 200 \
    --output inversion_result.json
```

This command runs the optimizer, refines both the shape and spin state, and
stores the recovered parameters in `inversion_result.json`.

## Extending the Model

The current implementation follows the spirit of the Kaasalainen convex inversion
method but keeps the math tractable by relying on finite-difference gradients
and a Lambertian scattering law. For improved fidelity you can:

- Replace the brightness model with more sophisticated scattering laws (e.g.
  Hapke or Lommel-Seeliger).
- Implement analytic gradients to accelerate convergence.
- Fuse multiple lightcurves acquired at different apparitions.
- Couple the inversion with adaptive mesh refinement for higher detail.

Contributions and adaptations are welcome!
