"""Linear tetrahedron-method integration of the number of states on a k-point grid.

Each parallelepiped cell of the periodic k-grid is split into 6 tetrahedra of equal
volume via a Kuhn triangulation along the geometrically shortest of the cell's 4
space diagonals (the conventional choice, which reduces interpolation error for
non-cubic grids). The number of states below a trial energy is then obtained by
summing the analytic per-tetrahedron cumulative DOS of Lehmann & Taut, phys. stat.
sol. (b) 54, 469 (1972).

This is the uncorrected ("linear") tetrahedron method: Bloechl's O(1/N^(2/3))
correction terms (Bloechl, Jepsen & Andersen, PRB 49, 16223 (1994)) are not applied.
For locating the Fermi energy the linear method is standard and sufficient.
"""

import numpy as np

_KUHN_PERMUTATIONS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def _shortest_diagonal_signs(g1: np.ndarray, g2: np.ndarray, g3: np.ndarray) -> tuple:
    """Signs (sx, sy, sz) in {+1, -1} such that sx*g1 + sy*g2 + sz*g3 is the
    shortest of the parallelepiped cell's 4 main space diagonals."""
    diagonals = {
        (1, 1, 1): g1 + g2 + g3,
        (-1, 1, 1): -g1 + g2 + g3,
        (1, -1, 1): g1 - g2 + g3,
        (1, 1, -1): g1 + g2 - g3,
    }
    return min(diagonals, key=lambda signs: np.linalg.norm(diagonals[signs]))


def _tetrahedron_corner_offsets(signs: tuple) -> np.ndarray:
    """(6, 4, 3) int array of 0/1 corner offsets within a grid cell, one row of 4
    corners per tetrahedron, Kuhn-triangulated along the diagonal given by `signs`."""
    offsets = np.zeros((6, 4, 3), dtype=int)
    for t, perm in enumerate(_KUHN_PERMUTATIONS):
        cur = [0, 0, 0]
        offsets[t, 0] = cur
        for v, axis in enumerate(perm, start=1):
            cur = list(cur)
            cur[axis] = 1
            offsets[t, v] = cur
    for axis in range(3):
        if signs[axis] == -1:
            offsets[:, :, axis] = 1 - offsets[:, :, axis]
    return offsets


def _corner_flat_indices(grid_shape: tuple, offsets: np.ndarray) -> np.ndarray:
    """Flat corner indices (into a C-order-raveled (nx, ny, nz) grid) for every cell
    and every tetrahedron corner, with periodic wraparound.

    :return: int array of shape (n_cells, 6, 4).
    """
    nx, ny, nz = grid_shape
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=-1)  # (n_cells, 3)

    corners = base[:, None, None, :] + offsets[None, :, :, :]  # (n_cells, 6, 4, 3)
    corners[..., 0] %= nx
    corners[..., 1] %= ny
    corners[..., 2] %= nz

    return (corners[..., 0] * ny + corners[..., 1]) * nz + corners[..., 2]


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), 0.0)


def _cumulative_fraction(sorted_corners: np.ndarray, energy: float) -> np.ndarray:
    """Fraction (in [0, 1]) of each tetrahedron's volume with (linearly interpolated)
    eigenvalue <= `energy`, given corner eigenvalues sorted ascending along the last
    axis (Lehmann & Taut, 1972).
    """
    e1 = sorted_corners[..., 0]
    e2 = sorted_corners[..., 1]
    e3 = sorted_corners[..., 2]
    e4 = sorted_corners[..., 3]

    e21, e31, e41 = e2 - e1, e3 - e1, e4 - e1
    e32, e42, e43 = e3 - e2, e4 - e2, e4 - e3

    seg1 = _safe_div((energy - e1) ** 3, e21 * e31 * e41)

    de2 = energy - e2
    term = e21**2 + 3 * e21 * de2 + 3 * de2**2 - _safe_div(e31 + e42, e32 * e42) * de2**3
    seg2 = _safe_div(term, e31 * e41)

    seg3 = 1.0 - _safe_div((e4 - energy) ** 3, e41 * e42 * e43)

    return np.where(
        energy < e1,
        0.0,
        np.where(energy < e2, seg1, np.where(energy < e3, seg2, np.where(energy < e4, seg3, 1.0))),
    )


def sorted_tetrahedron_corners(eigenvalues: np.ndarray, span_vectors: np.ndarray) -> np.ndarray:
    """Precompute sorted per-tetrahedron corner eigenvalues for every band.

    :param eigenvalues: shape ``(n_bands, nx, ny, nz)``, periodic grid (no duplicated
        boundary point, i.e. already trimmed as in
        :func:`fermisurface_utils.fermi.compute_fermi_energy_from_bxsf`).
    :param span_vectors: shape ``(3, 3)``, reciprocal lattice vectors as columns
        (same convention as :func:`fermisurface_utils.bxsf.read_bxsf`).
    :return: ndarray of shape ``(n_bands, n_cells * 6, 4)``, ascending-sorted along
        the last axis.
    """
    eigenvalues = np.asarray(eigenvalues)
    n_bands, nx, ny, nz = eigenvalues.shape
    span_vectors = np.asarray(span_vectors)

    g1 = span_vectors[:, 0] / nx
    g2 = span_vectors[:, 1] / ny
    g3 = span_vectors[:, 2] / nz
    signs = _shortest_diagonal_signs(g1, g2, g3)
    offsets = _tetrahedron_corner_offsets(signs)
    corner_flat_idx = _corner_flat_indices((nx, ny, nz), offsets)  # (n_cells, 6, 4)

    flat_eigenvalues = eigenvalues.reshape(n_bands, -1)  # (n_bands, n_cells)
    corners = flat_eigenvalues[:, corner_flat_idx]  # (n_bands, n_cells, 6, 4)
    corners = corners.reshape(n_bands, -1, 4)  # (n_bands, n_cells * 6, 4)
    corners.sort(axis=-1)
    return corners


def compute_n_electrons_tetrahedron(
    sorted_corners: np.ndarray, fermi_energy: float, prefactor: float = 2.0
) -> float:
    """Number of electrons with eigenvalue <= `fermi_energy`, from precomputed
    sorted tetrahedron corner eigenvalues (see :func:`sorted_tetrahedron_corners`).
    """
    n_tetra = sorted_corners.shape[1]
    fractions = _cumulative_fraction(sorted_corners, fermi_energy)
    return float(prefactor * fractions.sum() / n_tetra)
