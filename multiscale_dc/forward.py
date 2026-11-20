"""Exact gravity of rectangular prisms (Nagy et al., 2000), to make test data from a 3-D density model.

The synthetic datasets of the paper follow the same geometry: stations at the cell centres of the top face of
the model (z = 0), cells hanging down from there. With 50 m cubes and the density model ``m`` of a dataset
file, ``gz_prisms`` reproduces the stored anomaly ``d`` (in mGal).
"""
import numpy as np

G = 6.674e-11  # m^3 kg^-1 s^-2


def gz_prisms(xs, ys, zs, x1, x2, y1, y2, z1, z2, rho, chunk=2048):
    """Vertical gravity in mGal at stations (xs, ys, zs) of prisms [x1, x2] x [y1, y2] x [z1, z2] with density
    contrast ``rho`` (kg/m^3). All lengths in m, z positive DOWN (a station at height h has zs = -h).
    Positive contrasts below the stations give positive values."""
    xs, ys, zs = (np.asarray(a, dtype=float).ravel() for a in (xs, ys, zs))
    prisms = np.stack(np.broadcast_arrays(*(np.asarray(a, dtype=float).ravel()
                                            for a in (x1, x2, y1, y2, z1, z2, rho))), axis=1)
    out = np.zeros(xs.size)
    for c in range(0, len(prisms), chunk):
        p = prisms[c:c + chunk]
        acc = np.zeros((xs.size, len(p)))
        for i, xa in enumerate((p[:, 0], p[:, 1])):
            for j, ya in enumerate((p[:, 2], p[:, 3])):
                for k, za in enumerate((p[:, 4], p[:, 5])):
                    X, Y, Z = xa[None] - xs[:, None], ya[None] - ys[:, None], za[None] - zs[:, None]
                    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-12
                    acc += (-1) ** (i + j + k) * (X * np.log(Y + R) + Y * np.log(X + R) - Z * np.arctan2(X * Y, Z * R))
        out += (acc * p[None, :, 6]).sum(axis=1)
    return G * out * 1e5


def grid_gravity(model, cell, height=0.0, cell_z=None):
    """Gravity (mGal) of a density model on the grid of its cell centres.

    model   array (nz, ny, nx) of density contrasts in kg/m^3; layer 0 is at the top (depth 0 to cell_z).
    cell    horizontal cell size in m (also the grid spacing of the result).
    height  observation height in m above the top of the model.
    cell_z  vertical cell size in m (default: ``cell``).
    Returns an (ny, nx) array.
    """
    cell_z = cell if cell_z is None else cell_z
    nz, ny, nx = model.shape
    k, i, j = np.nonzero(model)
    xc, yc = (j + 0.5) * cell, (i + 0.5) * cell
    J, I = np.meshgrid(np.arange(nx), np.arange(ny))
    g = gz_prisms((J + 0.5) * cell, (I + 0.5) * cell, np.full(J.shape, -float(height)),
                  xc - cell / 2, xc + cell / 2, yc - cell / 2, yc + cell / 2, k * cell_z, (k + 1) * cell_z,
                  model[k, i, j])
    return g.reshape(ny, nx)
