"""Utilities for identifying and grouping periodic copies of isosurfaces."""

from typing import Any, Optional
from scipy.spatial import cKDTree, ConvexHull
from numpy.typing import NDArray
import numpy as np
from ifermi.surface import FermiSurface
from pymatgen.electronic_structure.core import Spin
from pymatgen.core.structure import Structure


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
    structure: Structure,
    atol: float = 1e-8,
) -> NDArray[np.bool_]:
    # hull.equations: a*x + b <= 0 for points inside the convex hull
    bz_facets = structure.lattice.reciprocal_lattice.get_wigner_seitz_cell()
    bz_vertices = np.unique(np.vstack(bz_facets), axis=0)
    bz_hull = ConvexHull(bz_vertices)
    return np.all(bz_hull.equations[:, :-1] @ points.T + bz_hull.equations[:, -1][:, None] <= atol, axis=0)


def supercell_fractional_bounds(
    kpoints: NDArray[np.float64],
    supercell_pad: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return lower/upper fractional bounds of an expanded supercell mesh.

    Args:
        kpoints: Fractional k-points used for the original (unexpanded) mesh.
        supercell_pad: Expansion amount applied on each side in fractional units.

    Returns:
        (bounds_min, bounds_max), each shape (3,).
    """
    bounds_min = np.min(kpoints, axis=0) - supercell_pad
    bounds_max = np.max(kpoints, axis=0) + supercell_pad
    return bounds_min, bounds_max


def surface_touches_supercell_boundary(
    vertices: NDArray[np.float64],
    reciprocal_lattice: NDArray[np.float64],
    bounds_min: NDArray[np.float64],
    bounds_max: NDArray[np.float64],
    atol: float = 1e-6,
) -> bool:
    """Check whether any isosurface vertex touches supercell outer bounds.

    Args:
        vertices: Isosurface vertices in reciprocal-cartesian coordinates.
        reciprocal_lattice: Reciprocal lattice matrix used to convert to fractional.
        bounds_min: Lower fractional bound of the expanded supercell.
        bounds_max: Upper fractional bound of the expanded supercell.
        atol: Absolute tolerance for boundary matching.

    Returns:
        True if any vertex is on any outer supercell boundary plane.
    """
    reciprocal_lattice_inv = np.linalg.inv(reciprocal_lattice)
    frac_vertices = vertices @ reciprocal_lattice_inv
    touches_min = np.isclose(frac_vertices, bounds_min[None, :], atol=atol)
    touches_max = np.isclose(frac_vertices, bounds_max[None, :], atol=atol)
    return bool(np.any(touches_min | touches_max))

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


def find_unique_surfaces(
    fs: FermiSurface,
    spin: Spin,
    structure: Structure,
    kpoints: Optional[NDArray[np.float64]] = None,
    discard_boundary_touching: bool = False,
    boundary_atol: float = 1e-6,
    atol: float = 2e-5,
    shift_tol: float = 1e-6,
    area_rtol: float = 1e-6,
) -> tuple[list[Any], list[int], list[int]]:
    """Select one representative surface per periodic-copy group.

    The representative is chosen from each periodic-copy group by preferring the
    surface with the most vertices inside the first Brillouin zone, then the
    largest area.

    If ``discard_boundary_touching`` is True, candidates touching the outer
    supercell boundary are skipped before selection.
    """
    groups, _ = find_periodic_copy_groups(
        fs,
        spin=spin,
        structure=structure,
        atol=atol,
        shift_tol=shift_tol,
        area_rtol=area_rtol,
    )

    surfaces = fs.isosurfaces[spin]
    bounds_min = bounds_max = None
    reciprocal_lattice = None
    if discard_boundary_touching:
        if kpoints is None:
            raise ValueError("kpoints is required when discard_boundary_touching=True")
        bounds_min, bounds_max = supercell_fractional_bounds(kpoints, supercell_pad=1.0)
        reciprocal_lattice = structure.lattice.reciprocal_lattice.matrix

    unique_surfaces = []
    unique_indices = []
    discarded_boundary_indices = []

    for group in groups:
        candidates = []
        for idx in group:
            surface = surfaces[idx]
            if discard_boundary_touching and surface_touches_supercell_boundary(
                surface.vertices,
                reciprocal_lattice,
                bounds_min,
                bounds_max,
                atol=boundary_atol,
            ):
                discarded_boundary_indices.append(idx)
                continue

            inside_mask = points_in_first_bz(surface.vertices, structure)
            n_inside = int(inside_mask.sum())
            if n_inside > 0:
                candidates.append((n_inside, surface.area, idx, surface))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            unique_surfaces.append(candidates[0][3])
            unique_indices.append(candidates[0][2])

    return unique_surfaces, unique_indices, sorted(set(discarded_boundary_indices))
