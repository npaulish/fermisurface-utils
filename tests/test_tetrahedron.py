"""Tests for the linear tetrahedron-method Fermi energy integration."""

import numpy as np
import pytest

from fermisurface_utils.bxsf import write_bxsf
from fermisurface_utils.fermi import (
    FermiEnergyNotFoundError,
    compute_fermi_energy,
    compute_fermi_energy_from_bxsf,
    compute_fermi_energy_tetrahedron,
)
from fermisurface_utils.tetrahedron import (
    _cumulative_fraction,
    _shortest_diagonal_signs,
    _tetrahedron_corner_offsets,
)


def _tet_volume(v0, v1, v2, v3):
    return abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0


@pytest.mark.parametrize("signs", [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)])
def test_kuhn_triangulation_tiles_unit_cube(signs):
    offsets = _tetrahedron_corner_offsets(signs).astype(float)  # (6, 4, 3)
    volumes = [_tet_volume(*offsets[t]) for t in range(6)]
    assert all(v == pytest.approx(1 / 6) for v in volumes)
    assert sum(volumes) == pytest.approx(1.0)


def test_shortest_diagonal_selection():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g1, g2, g3 = rng.normal(size=(3, 3))
        diagonals = {
            (1, 1, 1): g1 + g2 + g3,
            (-1, 1, 1): -g1 + g2 + g3,
            (1, -1, 1): g1 - g2 + g3,
            (1, 1, -1): g1 + g2 - g3,
        }
        signs = _shortest_diagonal_signs(g1, g2, g3)
        chosen_norm = np.linalg.norm(diagonals[signs])
        assert all(chosen_norm <= np.linalg.norm(v) + 1e-12 for v in diagonals.values())


def test_cumulative_fraction_matches_monte_carlo():
    rng = np.random.default_rng(1)
    for _ in range(8):
        corners = np.sort(rng.uniform(-5, 5, size=4))
        e1, e2, e3, e4 = corners
        bary = rng.dirichlet([1, 1, 1, 1], size=300_000)
        values = bary @ corners
        for energy in np.linspace(e1 - 1, e4 + 1, 7):
            analytic = float(_cumulative_fraction(corners, energy))
            monte_carlo = np.mean(values <= energy)
            assert analytic == pytest.approx(monte_carlo, abs=6e-3)


def test_degenerate_flat_tetrahedron_is_step_function():
    flat = np.full(4, 3.0)
    assert _cumulative_fraction(flat, 2.9) == 0.0
    assert _cumulative_fraction(flat, 3.1) == 1.0


def test_flat_band_unachievable_filling_raises():
    # A perfectly flat band has a delta-function DOS: no target strictly between
    # 0 and 2 electrons is achievable at zero broadening.
    n = 6
    eigenvalues = np.full((1, n, n, n), 3.0)
    span_vectors = np.eye(3)
    with pytest.raises(FermiEnergyNotFoundError):
        compute_fermi_energy_tetrahedron(eigenvalues, span_vectors, num_electrons=1.0)


def test_filled_flat_band_plus_dispersive_band():
    n = 8
    kx, ky, kz = np.meshgrid(np.arange(n) / n, np.arange(n) / n, np.arange(n) / n, indexing="ij")
    dispersive = -2.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz))
    eigenvalues = np.stack([np.full((n, n, n), -10.0), dispersive], axis=0)
    span_vectors = np.eye(3)

    fermi_energy, _tol = compute_fermi_energy_tetrahedron(
        eigenvalues, span_vectors, num_electrons=3.0
    )
    assert dispersive.min() < fermi_energy < dispersive.max()


def test_tight_binding_quarter_filling_matches_symmetry_point():
    # E(k) = -2(cos kx + cos ky + cos kz) is particle-hole symmetric about E=0,
    # so quarter filling (1 electron out of 2, prefactor=2) must sit at E_F=0.
    n = 10
    kx, ky, kz = np.meshgrid(np.arange(n) / n, np.arange(n) / n, np.arange(n) / n, indexing="ij")
    eigenvalues = (-2.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz)))[
        None, :, :, :
    ]
    span_vectors = 2 * np.pi * np.eye(3)

    fermi_energy, _tol = compute_fermi_energy_tetrahedron(
        eigenvalues, span_vectors, num_electrons=1.0
    )
    assert fermi_energy == pytest.approx(0.0, abs=1e-6)


def test_tetrahedron_and_pointwise_agree_in_the_same_ballpark():
    n = 10
    kx, ky, kz = np.meshgrid(np.arange(n) / n, np.arange(n) / n, np.arange(n) / n, indexing="ij")
    eigenvalues = (-2.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz)))[
        None, :, :, :
    ]
    span_vectors = 2 * np.pi * np.eye(3)

    fe_tetra, _ = compute_fermi_energy_tetrahedron(eigenvalues, span_vectors, num_electrons=1.0)

    eigenvalues_flat = eigenvalues.reshape(1, -1).T
    fe_step, _ = compute_fermi_energy(
        eigenvalues_flat, num_electrons=1.0, kBT=0.0, smearing="none"
    )

    assert eigenvalues.min() < fe_tetra < eigenvalues.max()
    assert abs(fe_tetra - fe_step) < 1.0


def test_compute_fermi_energy_from_bxsf_tetrahedron_matches_direct_call(tmp_path):
    n = 10
    kx, ky, kz = np.meshgrid(np.arange(n) / n, np.arange(n) / n, np.arange(n) / n, indexing="ij")
    band = -2.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz))
    span_vectors = 2 * np.pi * np.eye(3)

    # BXSF's "general grid" duplicates the boundary point along each axis.
    E = np.zeros((1, n + 1, n + 1, n + 1))
    E[0, :n, :n, :n] = band
    E[0, n, :, :] = E[0, 0, :, :]
    E[0, :, n, :] = E[0, :, 0, :]
    E[0, :, :, n] = E[0, :, :, 0]

    bxsf_path = tmp_path / "test.bxsf"
    write_bxsf(str(bxsf_path), fermi_energy=0.0, origin=[0, 0, 0], span_vectors=span_vectors, E=E)

    result = compute_fermi_energy_from_bxsf(str(bxsf_path), num_electrons=1.0, smearing="tetrahedron")
    direct_fermi_energy, _tol = compute_fermi_energy_tetrahedron(
        band[None, :, :, :], span_vectors, num_electrons=1.0
    )
    assert result.fermi_energy == pytest.approx(direct_fermi_energy, abs=1e-9)
    assert result.fermi_energy == pytest.approx(0.0, abs=1e-6)


def test_skewed_reciprocal_lattice_gives_bracketed_fermi_energy():
    span_vectors = np.array(
        [
            [1.0, 0.3, 0.1],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0],
        ]
    )
    n = 5
    rng = np.random.default_rng(2)
    eigenvalues = rng.uniform(-1, 1, size=(2, n, n, n))

    fermi_energy, _tol = compute_fermi_energy_tetrahedron(
        eigenvalues, span_vectors, num_electrons=1.3
    )
    assert eigenvalues.min() < fermi_energy < eigenvalues.max()
