"""Tests for the certificate checks, the Mode II-B helpers and the public API.

Complements ``test_mpu_generators.py``: negative tests (every algebraic
check must reject a perturbed tensor), the rectangular intertwiner of the
thesis Lee-Yang example, and the named-family registry.
"""

import numpy as np
import pytest
from TreeMPU.checks import dense_unitary_from_tensors

from mpu_generators import FAMILIES, MPUGenerator, UniformMPUGenerator, build_family
from mpu_generators.base import (
    _check_fusion_closure,
    _check_group_boundary,
    _check_nested_flag_closure,
    _check_row_resolution,
    _finite_group_table,
    _verify_chain,
)
from mpu_generators.skeletons import (
    lee_yang_skeleton,
    mpo_product_tensor,
    solve_intertwiner,
    verify_intertwiner,
)

RNG = np.random.default_rng(1234)


def _perturb(A, eps=1e-3):
    return A + eps * (RNG.standard_normal(A.shape) + 1j * RNG.standard_normal(A.shape))


# ---------------------------------------------------------------------------
# Negative tests: every check must reject a perturbed object
# ---------------------------------------------------------------------------


def test_row_resolution_check_rejects_perturbed_tensor():
    t, _, _ = UniformMPUGenerator(seed=0).branching(d=2, bond_dim=3, n_sites=2)
    _check_row_resolution(t[0])
    with pytest.raises(RuntimeError):
        _check_row_resolution(_perturb(t[0]))


def test_fusion_closure_checks_reject_perturbed_channels():
    from mpu_generators.base import _cyclic_character_channels, _nested_flag_channels

    y = _cyclic_character_channels(3, 3, np.random.default_rng(0))
    _check_fusion_closure(y)
    with pytest.raises(RuntimeError):
        _check_fusion_closure(_perturb(y))
    y = _nested_flag_channels(3, 3, np.random.default_rng(0))
    _check_nested_flag_closure(y)
    with pytest.raises(RuntimeError):
        _check_nested_flag_closure(_perturb(y))


def test_group_boundary_check_rejects_perturbed_boundary():
    from mpu_generators.base import _group_fourier_boundary

    elements, mult, _inv, identity, irreps = _finite_group_table("S3")
    idx = {g: i for i, g in enumerate(elements)}
    l = _group_fourier_boundary(elements, irreps, np.random.default_rng(0))
    _check_group_boundary(l, elements, mult, idx, identity)
    with pytest.raises(RuntimeError):
        _check_group_boundary(_perturb(l), elements, mult, idx, identity)


def test_verify_chain_rejects_non_unitary_chain_with_correct_normalization():
    """diag(sqrt(2), 0) has Tr(U^dag U)/d = 1 but is not unitary: the
    normalization alone passes it, the dense check must not."""
    A = np.zeros((2, 2, 1, 1), dtype=complex)
    A[0, 0, 0, 0] = np.sqrt(2)  # one site: diag(sqrt 2, 0), Tr(D^dag D)/d = 1
    one = np.ones(1, dtype=complex)
    with pytest.raises(RuntimeError, match=r"U\^dag U - I"):
        _verify_chain([A, A], one, one)


# ---------------------------------------------------------------------------
# Mode II-B helpers
# ---------------------------------------------------------------------------


def test_mpo_product_tensor_reduces_to_matrix_product_at_bond_one():
    X = RNG.standard_normal((3, 3)) + 1j * RNG.standard_normal((3, 3))
    Y = RNG.standard_normal((3, 3)) + 1j * RNG.standard_normal((3, 3))
    P = mpo_product_tensor(X.T[:, :, None, None], Y.T[:, :, None, None])
    np.testing.assert_allclose(P[:, :, 0, 0].T, X @ Y, atol=1e-12)


def test_lee_yang_rectangular_intertwiner_exists():
    """Thesis eq:intertwiner: sigma*sigma = e + sigma needs an injective
    V : C^(2+3) -> C^9 (dimensions differ)."""
    sk = lee_yang_skeleton()
    Bss = mpo_product_tensor(sk.channels["sigma"], sk.channels["sigma"])
    V = solve_intertwiner(Bss, [sk.channels["e"], sk.channels["sigma"]])
    assert V.shape == (9, 5)
    assert np.linalg.matrix_rank(V) == 5
    assert verify_intertwiner(Bss, [sk.channels["e"], sk.channels["sigma"]], V) < 1e-10


def test_solve_intertwiner_rejects_impossible_dimensions():
    B = np.zeros((2, 2, 2, 2), dtype=complex)
    with pytest.raises(ValueError):
        solve_intertwiner(B, [np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 1, 1))])


@pytest.mark.parametrize("alpha,beta", [(0.0, np.pi), (0.3, 1.7), (-1.2, 0.4)])
def test_lee_yang_boundary_family_is_unitary(alpha, beta):
    t, l, r = UniformMPUGenerator(seed=0, verify=False).fusion_bond_resolved(
        lee_yang_skeleton(alpha, beta), n_sites=3
    )
    U = dense_unitary_from_tensors(t, l, r)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


# ---------------------------------------------------------------------------
# Family registry and reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_family_builds_a_unitary(name):
    kwargs = {"fusion_matrix_units": {"d": 2, "bond_dim": 2}}.get(name, {"d": 2, "bond_dim": 2})
    t, l, r = build_family(name, seed=0, **kwargs)
    dim = int(np.prod([A.shape[0] for A in t]))
    if dim > 2**10:
        pytest.skip("too large for a dense check")
    U = dense_unitary_from_tensors(t, l, r)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(dim), atol=1e-9)


def test_uniform_site_tensor_is_independent_of_n_sites():
    """The ensembles of the paper rely on this (one tensor per seed)."""
    a = UniformMPUGenerator(seed=7).branching(d=2, bond_dim=2, n_sites=2)[0][0]
    b = UniformMPUGenerator(seed=7).branching(d=2, bond_dim=2, n_sites=8)[0][0]
    np.testing.assert_array_equal(a, b)


def test_repeated_calls_are_reproducible():
    g = MPUGenerator(seed=3)
    t1, l1, _r1 = g.fusion(d=3, bond_dim=3, n_sites=3, skeleton="group", group="Zn", group_n=3)
    t2, l2, _r2 = g.fusion(d=3, bond_dim=3, n_sites=3, skeleton="group", group="Zn", group_n=3)
    for a, b in zip(t1, t2):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(l1, l2)


def test_cnot_witness_is_the_group_automaton_Z2():
    """group_automaton("Zn", 2) with l = e_0 is the CNOT-staircase witness:
    the N = 2 chain is a CNOT (control on one site, target on the other)."""
    A, l, r = UniformMPUGenerator(seed=0).group_automaton(group="Zn", n=2)
    U = dense_unitary_from_tensors([A, A], l, r)
    # U is a permutation matrix of order 2 acting non-trivially on 2 qubits
    assert np.allclose(U @ U, np.eye(4))
    assert np.allclose(np.abs(U), np.round(np.abs(U)))
    assert not np.allclose(U, np.eye(4))


def test_explicit_blocking_is_refused_before_allocation():
    with pytest.raises(ValueError, match="safety cap"):
        MPUGenerator(seed=0).branching(d=2, bond_dim=2, n_blocks=2, blocking=16)
