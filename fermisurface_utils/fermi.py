"""Compute the Fermi energy from interpolated band eigenvalues on a k-point grid.

Python port of the Wannier.jl bisection algorithm
(https://github.com/qiaojunfeng/Wannier.jl/blob/main/src/interpolation/fermi_energy.jl)
and the ``compute_Fermi.jl`` CLI driver
(https://github.com/npaulish/aiida-skeaf/tree/compute_Fermi/utils/computeFermi),
operating directly on the eigenvalue grid stored in a .bxsf file.
"""

from dataclasses import dataclass
import typing as ty

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from .bxsf import read_bxsf


class FermiEnergyNotFoundError(RuntimeError):
    """Raised when the Fermi energy could not be bracketed or converged."""


def _occupation_none(x: np.ndarray) -> np.ndarray:
    # x == 0 (state exactly at the Fermi level) gives 0 * inf = nan upstream,
    # and nan > 0 is False, so it is counted as occupied, matching the Julia
    # `x > 0 ? zero(x) : one(x)` behaviour.
    with np.errstate(invalid="ignore"):
        return np.where(x > 0, 0.0, 1.0)


def _occupation_fermi_dirac(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(x))


def _occupation_cold(x: np.ndarray) -> np.ndarray:
    """Marzari-Vanderbilt cold smearing."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    return -erf(x + inv_sqrt2) / 2 + 1 / np.sqrt(2 * np.pi) * np.exp(-((x + inv_sqrt2) ** 2)) + 0.5


_SMEARING_FUNCTIONS = {
    "none": _occupation_none,
    "fermi-dirac": _occupation_fermi_dirac,
    "fd": _occupation_fermi_dirac,
    "marzari-vanderbilt": _occupation_cold,
    "cold": _occupation_cold,
}


def occupation(
    eigenvalues: np.ndarray,
    fermi_energy: float,
    kBT: float,
    smearing: str,
    prefactor: float = 2,
) -> np.ndarray:
    """Occupation numbers for the given eigenvalues, Fermi energy and smearing.

    :param eigenvalues: eigenvalues in eV, any shape.
    :param fermi_energy: Fermi energy in eV.
    :param kBT: smearing width in eV; 0 means a sharp step (no smearing).
    :param smearing: one of "none", "fermi-dirac"/"fd", "marzari-vanderbilt"/"cold".
    :param prefactor: occupation prefactor, 2 for non-SOC, 1 for SOC.
    """
    occ_fn = _SMEARING_FUNCTIONS[smearing]
    inv_kBT = np.inf if kBT == 0 else 1.0 / kBT
    x = (np.asarray(eigenvalues) - fermi_energy) * inv_kBT
    return prefactor * occ_fn(x)


def compute_n_electrons(
    eigenvalues: np.ndarray,
    fermi_energy: float,
    kBT: float,
    smearing: str,
    prefactor: float = 2,
    kweights: ty.Optional[np.ndarray] = None,
) -> float:
    """Number of electrons for the given eigenvalues, Fermi energy and smearing.

    :param eigenvalues: eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    :param kweights: per-kpoint weight, shape ``(n_kpoints,)``, summing to 1.
        Defaults to a uniform ``1 / n_kpoints`` weight.
    """
    occ = occupation(eigenvalues, fermi_energy, kBT, smearing, prefactor)
    n_kpoints = occ.shape[0]
    if kweights is None:
        kweights = np.full(n_kpoints, 1.0 / n_kpoints)
    return float(np.sum(np.asarray(kweights)[:, None] * occ))


def compute_fermi_energy(
    eigenvalues: np.ndarray,
    num_electrons: float,
    kBT: float = 0.0,
    smearing: str = "none",
    prefactor: float = 2,
    kweights: ty.Optional[np.ndarray] = None,
    tol_n_electrons: float = 1e-6,
) -> ty.Tuple[float, float]:
    """Find the Fermi energy for which the occupied states sum to ``num_electrons``.

    The Fermi energy is bracketed between the min/max eigenvalues and located
    with Brent's method. If the achieved electron count is not within
    ``tol_n_electrons``, the tolerance is doubled and re-checked, up to
    ``max(1e-3, tol_n_electrons)``, mirroring the retry logic in
    ``compute_Fermi.jl`` (needed there because of the coarser tolerance used
    by Roots.jl's plain bisection; kept here for parity even though
    ``scipy.optimize.brentq`` converges to a much tighter tolerance already).

    :return: ``(fermi_energy, tol_n_electrons_used)``.
    """
    eigenvalues = np.asarray(eigenvalues)

    def excess(fermi_energy):
        return (
            compute_n_electrons(eigenvalues, fermi_energy, kBT, smearing, prefactor, kweights)
            - num_electrons
        )

    min_e = eigenvalues.min() - 1
    max_e = eigenvalues.max() + 1
    excess_min, excess_max = excess(min_e), excess(max_e)
    if not excess_min <= 0 <= excess_max:
        raise FermiEnergyNotFoundError(
            f"Fermi energy not bracketed: "
            f"excess({min_e})={excess_min}, excess({max_e})={excess_max}"
        )

    fermi_energy = brentq(excess, min_e, max_e)
    residual = abs(excess(fermi_energy))

    tol_upperbound = max(1e-3, tol_n_electrons)
    tol = tol_n_electrons
    while residual > tol:
        tol *= 2
        if tol > tol_upperbound:
            raise FermiEnergyNotFoundError(
                f"Failed to find Fermi energy within tolerance: residual={residual}, "
                f"tol_n_electrons_upperbound={tol_upperbound}"
            )

    return fermi_energy, tol


@dataclass
class FermiEnergyResult:
    """Result of :func:`compute_fermi_energy_from_bxsf`."""

    fermi_energy: float
    tol_n_electrons: float
    eigenvalue_below: float
    eigenvalue_above: float
    bandgap: float
    bands_crossing_fermi: ty.List[int]
    band_min_max: ty.List[ty.Tuple[float, float]]
    n_bands: int
    grid_shape: ty.Tuple[int, int, int]


def compute_fermi_energy_from_bxsf(
    bxsf_path: str,
    num_electrons: float,
    kBT: float = 0.0,
    smearing: str = "none",
    prefactor: float = 2,
    tol_n_electrons: float = 1e-6,
) -> FermiEnergyResult:
    """Compute the Fermi energy from the interpolated bands in a .bxsf file.

    Per the BXSF spec, the eigenvalue grid is a "general grid": the points on
    the right BZ boundary duplicate the ones on the left along each axis.
    These duplicates are dropped before integrating the DOS, otherwise the
    computed Fermi energy is biased (band min/max, used for
    ``bands_crossing_fermi``, are unaffected by the duplicates so the full
    grid is used for those).
    """
    _fermi_energy_bxsf, _origin, _span_vectors, _X, _Y, _Z, E = read_bxsf(bxsf_path)

    n_bands, n_x, n_y, n_z = E.shape
    eigenvalues = E[:, :-1, :-1, :-1].reshape(n_bands, -1).T

    fermi_energy, tol_used = compute_fermi_energy(
        eigenvalues, num_electrons, kBT, smearing, prefactor, tol_n_electrons=tol_n_electrons
    )

    flat = eigenvalues.ravel()
    below = flat[flat < fermi_energy]
    above = flat[flat > fermi_energy]
    if below.size == 0 or above.size == 0:
        # The bisection returned a Fermi energy at (or beyond) the edge of the
        # interpolated band window, so there is no state on one side of it and the
        # gap is undefined. The Fermi energy itself is unreliable here too: the
        # requested electron count is only satisfied at the very edge of the window,
        # which usually means the bxsf does not contain enough bands.
        side = "below" if below.size == 0 else "above"
        raise FermiEnergyNotFoundError(
            f"Fermi energy {fermi_energy} has no eigenvalue {side} it within the "
            f"interpolated range [{float(flat.min())}, {float(flat.max())}]; "
            "the bxsf band window does not bracket the requested electron count"
        )
    eigenvalue_below = float(below.max())
    eigenvalue_above = float(above.min())

    band_min_max = [(float(E[ib].min()), float(E[ib].max())) for ib in range(n_bands)]
    bands_crossing_fermi = [
        ib + 1 for ib, (bmin, bmax) in enumerate(band_min_max) if bmin <= fermi_energy <= bmax
    ]

    return FermiEnergyResult(
        fermi_energy=fermi_energy,
        tol_n_electrons=tol_used,
        eigenvalue_below=eigenvalue_below,
        eigenvalue_above=eigenvalue_above,
        bandgap=eigenvalue_above - eigenvalue_below,
        bands_crossing_fermi=bands_crossing_fermi,
        band_min_max=band_min_max,
        n_bands=n_bands,
        grid_shape=(n_x, n_y, n_z),
    )
