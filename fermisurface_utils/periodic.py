"""Utilities for identifying and grouping periodic copies of isosurfaces."""

from typing import Any, Optional
from scipy.spatial import cKDTree, ConvexHull
from numpy.typing import NDArray
import numpy as np
from ifermi.surface import FermiSurface
from pymatgen.electronic_structure.core import Spin


def _periodic_copy_kdtree_vertices(
        va: NDArray[np.float64],
        vb: NDArray[np.float64],
        B: NDArray[np.float64],
        Binv: NDArray[np.float64],
        atol: float = 2e-5,
        shift_tol: float = 1e-6,
    ) -> tuple[bool, Optional[NDArray[np.int_]], dict[str, Any]]:
    """Check if vb is a periodic copy of va; return (ok, n_shift, info)."""
    if va.shape != vb.shape:
        return False, None, {"reason": "shape_mismatch"}

    # Estimate integer reciprocal-lattice shift from centroids.
    ca_frac = va.mean(axis=0) @ Binv
    cb_frac = vb.mean(axis=0) @ Binv
    d_frac = cb_frac - ca_frac
    n = np.rint(d_frac).astype(int)

    if np.linalg.norm(d_frac - n) > shift_tol:
        return False, n, {"reason": "non_integer_shift", "d_frac": d_frac}

    # Translate A and compare point clouds order-independently.
    va_t = va + (n @ B)

    tree_b = cKDTree(vb)
    d_ab, _ = tree_b.query(va_t, k=1)

    tree_a = cKDTree(va_t)
    d_ba, _ = tree_a.query(vb, k=1)

    ok = (d_ab.max() <= atol) and (d_ba.max() <= atol)
    info = {
        "max_dist_ab": float(d_ab.max()),
        "max_dist_ba": float(d_ba.max()),
        "n_bad_ab": int((d_ab > atol).sum()),
        "n_bad_ba": int((d_ba > atol).sum()),
    }
    return ok, n, info

def points_in_first_bz(
    points: NDArray[np.float64],
    hull: ConvexHull,
    atol: float = 1e-8,
) -> NDArray[np.bool_]:
    # hull.equations: a*x + b <= 0 for points inside the convex hull
    return np.all(hull.equations[:, :-1] @ points.T + hull.equations[:, -1][:, None] <= atol, axis=0)

def find_periodic_copy_groups(
        fs: FermiSurface,
        spin: Spin,
        structure: Any,
        atol: float = 2e-5,
        shift_tol: float = 1e-6,
        area_rtol: float = 1e-6,
    ) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """
    Find groups of isosurface indices that are periodic copies of each other.

    Returns:
        groups: list[list[int]] connected components of periodic-copy graph,
                sorted by max isosurface area in each group (largest first)
        pair_matches: list[dict] pairwise matches with shift vectors
    """
    surfaces = fs.isosurfaces[spin]
    n_surfaces = len(surfaces)

    B = structure.lattice.reciprocal_lattice.matrix
    Binv = np.linalg.inv(B)

    vertices = [s.vertices for s in surfaces]
    band_idx = np.array([s.band_idx for s in surfaces])
    areas = np.array([s.area for s in surfaces], dtype=float)

    # Build graph of periodic-copy relations.
    adjacency = [set() for _ in range(n_surfaces)]
    pair_matches = []

    for i in range(n_surfaces):
        for j in range(i + 1, n_surfaces):
            # Cheap filters first.
            if band_idx[i] != band_idx[j]:
                continue

            ai = areas[i]
            aj = areas[j]
            if not np.isclose(ai, aj, rtol=area_rtol, atol=0.0):
                continue

            ok, n_shift, info = _periodic_copy_kdtree_vertices(
                vertices[i], vertices[j], B=B, Binv=Binv, atol=atol, shift_tol=shift_tol
            )
            if ok:
                adjacency[i].add(j)
                adjacency[j].add(i)
                pair_matches.append(
                    {
                        "i": i,
                        "j": j,
                        "band_idx": int(band_idx[i]),
                        "area": float(ai),
                        "shift": tuple(int(x) for x in n_shift),
                        "max_dist_ab": info["max_dist_ab"],
                        "max_dist_ba": info["max_dist_ba"],
                    }
                )

    # Connected components = equivalence classes.
    groups = []
    visited = np.zeros(n_surfaces, dtype=bool)

    for start in range(n_surfaces):
        if visited[start]:
            continue
        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        groups.append(sorted(comp))

    # Largest-area groups first; tie-break on group size.
    groups = sorted(groups, key=lambda g: (max(areas[idx] for idx in g), len(g)), reverse=True)

    return groups, pair_matches
