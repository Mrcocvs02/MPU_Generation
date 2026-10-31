"""Shared-basis counter automata and Mode II ``cyclic`` with ``bond_dim > d``.

The automata are Mode I tensors whose vertices read the site in one common
basis. The tests check exact unitarity for several choices, the exact
identification with the earlier ``group_automaton`` and with Mode II
``cyclic`` (``bond_dim == d``), the persistent operator tail that
distinguishes them from Haar-basis rings, and invalid input. For
``fusion(skeleton="cyclic")`` with ``bond_dim > d`` they check unitarity (also
site dependent), the commutation with the total charge, the operator-Schmidt
rank, and the automatic blocking.
"""

import warnings

import numpy as np
import pytest
from TreeMPU.checks import dense_unitary_from_tensors
from TreeMPU.utils import satisfies_assumption_1

from mpu_generators import MPUGenerator, UniformMPUGenerator


def _unitary_error(tensors, l, r):
    """Return the Frobenius norm of ``U^dag U - 1`` for the dense chain."""
    U = dense_unitary_from_tensors(tensors, l, r)
    return np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))


def _tail_weight(U, n, d, site, P, radius):
    """Fraction of ``U P_site U^dag`` not supported in ``[site-radius, site+radius]``.

    ``U`` is little-endian (site 0 least significant), as returned by
    ``dense_unitary_from_tensors``.
    """
    op = U @ np.kron(np.kron(np.eye(d ** (n - site - 1)), P), np.eye(d**site)) @ U.conj().T
    keep = [n - 1 - k for k in range(max(0, site - radius), min(n - 1, site + radius) + 1)]
    out = [a for a in range(n) if a not in keep]
    t = op.reshape([d] * (2 * n)).transpose(
        keep + out + [n + a for a in keep] + [n + a for a in out]
    )
    t = t.reshape(d ** len(keep), d ** len(out), d ** len(keep), d ** len(out))
    reduced = np.einsum("aobo->ab", t) / d ** len(out)
    return 1 - np.linalg.norm(reduced) ** 2 * d ** len(out) / np.linalg.norm(op) ** 2


def _operator_schmidt_rank(U, n, d):
    """Number of non-zero operator-Schmidt values of ``U`` at the middle cut."""
    h = n // 2
    t = U.reshape([d] * (2 * n))
    left = list(range(n - h, n)) + [n + a for a in range(n - h, n)]  # sites 0..h-1, little-endian
    right = [a for a in range(2 * n) if a not in left]
    m = t.transpose(left + right).reshape(d ** (2 * h), d ** (2 * (n - h)))
    return int((np.linalg.svd(m, compute_uv=False) > 1e-9).sum())


PAULI_X = np.array([[0, 1.0], [1.0, 0]])
PAULI_Z = np.diag([1.0, -1.0])


# --- shared_basis_automaton / cayley_automaton ---------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "kw, n",
    [
        ({"D": 4, "d": 2}, 7),
        ({"D": 3, "d": 3}, 5),
        ({"transitions": [[0, 0, 1], [2, 1, 1]]}, 7),  # not permutations: still unitary
        ({"transitions": [[1, 0, 2], [1, 2, 0], [0, 1, 2]], "start": 2}, 5),
    ],
)
def test_shared_basis_automaton_is_unitary(kw, n, seed):
    """Unitarity holds for any maps, Haar R and V, and random final phases."""
    gen = UniformMPUGenerator(seed=seed)
    phases = np.random.default_rng(seed).uniform(
        0, 2 * np.pi, 4 if "D" in kw and kw["D"] == 4 else 3
    )
    for final in (None, phases[: (kw.get("D") or len(kw["transitions"][0]))]):
        A, l, r = gen.shared_basis_automaton(final_phases=final, **kw)
        assert _unitary_error([A] * n, l, r) < 1e-10


def test_shared_basis_automaton_shares_the_input_projectors():
    """Every vertex reads in the same basis: ``d`` distinct input projectors in all."""
    A, _, _ = UniformMPUGenerator(seed=0).shared_basis_automaton(D=5, d=3)
    axes = []
    for v in range(5):
        for u in range(5):
            block = A[:, :, v, u].T  # operator, out <- in
            if np.linalg.norm(block) > 1e-9:
                pi = block.conj().T @ block
                if not any(np.linalg.norm(pi - q) < 1e-9 for q in axes):
                    axes.append(pi)
    assert len(axes) == 3


def test_shared_basis_automaton_defaults_and_reproducibility():
    """Same seed, same tensor; the seed argument overrides the generator seed."""
    A1 = UniformMPUGenerator(seed=4).shared_basis_automaton(D=4)[0]
    A2 = UniformMPUGenerator(seed=0).shared_basis_automaton(D=4, seed=4)[0]
    A3 = UniformMPUGenerator(seed=5).shared_basis_automaton(D=4)[0]
    assert np.array_equal(A1, A2)
    assert not np.allclose(A1, A3)
    _, l, r = UniformMPUGenerator().shared_basis_automaton(D=4, start=2)
    assert np.array_equal(l, np.eye(4)[2]) and np.allclose(r, 1)


@pytest.mark.parametrize("group, n", [("Zn", 2), ("Zn", 3), ("Zn", 4), ("Sn", 3)])
def test_group_automaton_is_the_shared_basis_case(group, n):
    """``group_automaton`` = reading basis 1, letters = all elements, ``V_g`` = left product."""
    gen = UniformMPUGenerator()
    A_ref, l_ref, r_ref = gen.group_automaton(group, n)
    d = A_ref.shape[0]
    if group == "Zn":
        elements = list(range(n))

        def mult(g, h):
            return (g + h) % n

        identity = 0
    else:
        from itertools import permutations

        elements = list(permutations(range(n)))

        def mult(g, h):
            return tuple(g[h[i]] for i in range(n))

        identity = tuple(range(n))
    idx = {g: i for i, g in enumerate(elements)}
    transitions = [[idx[mult(g, x)] for g in elements] for x in elements]
    V = []
    for g in elements:
        perm = np.zeros((d, d))
        for x in elements:
            perm[idx[mult(g, x)], idx[x]] = 1.0
        V.append(perm)
    A, l, r = gen.shared_basis_automaton(
        transitions=transitions, R=np.eye(d), V=V, start=idx[identity]
    )
    assert np.allclose(A, A_ref, atol=1e-14)
    assert np.allclose(l, l_ref) and np.allclose(r, r_ref)


@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_mode2_cyclic_is_the_shared_basis_case(m, seed):
    """Mode II ``cyclic`` (``bond_dim == d``) is ``R exp(i phi(Q)) R^dag``: identity V, final phases."""
    tensors, l, r = UniformMPUGenerator(seed=seed).fusion(
        d=m, bond_dim=m, n_sites=4, skeleton="cyclic"
    )
    A_cyc = tensors[0]
    values, vectors = np.linalg.eig(A_cyc[:, :, 1, 1].T)  # y_1 = R diag(omega^k) R^dag
    k = np.round(np.angle(values) / (2 * np.pi / m)).astype(int) % m
    R, _ = np.linalg.qr(vectors[:, np.argsort(k)])
    f = np.array([sum(l[g] * np.exp(2j * np.pi * g * q / m) for g in range(m)) for q in range(m)])
    A, l2, r2 = UniformMPUGenerator().shared_basis_automaton(
        D=m, d=m, R=R, V=[np.eye(m)] * m, final_phases=np.angle(f)
    )
    n = 5
    U_cyc = dense_unitary_from_tensors([A_cyc] * n, l, r)
    U = dense_unitary_from_tensors([A] * n, l2, r2)
    overlap = np.vdot(U_cyc, U)
    assert np.linalg.norm(U_cyc - U / (overlap / abs(overlap))) < 1e-9


def test_cayley_automaton_s3_with_two_letters():
    """S3 with a transposition and a 3-cycle: d = 2 letters, D = 6 vertices, unitary."""
    gen = UniformMPUGenerator(seed=0)
    A, l, r = gen.cayley_automaton("S3")
    assert A.shape == (2, 2, 6, 6)
    assert np.array_equal(l, np.eye(6)[0])  # elements[0] is the identity
    assert _unitary_error([A] * 7, l, r) < 1e-10
    # every vertex is reachable from the identity (the two generators generate S3)
    reach = {0}
    for _ in range(6):
        reach |= {u for v in reach for u in range(6) if np.any(A[:, :, v, u])}
    assert reach == set(range(6))


def test_cayley_automaton_explicit_generators_and_validation():
    """Explicit generators are accepted; missing or foreign ones are refused."""
    gen = UniformMPUGenerator(seed=1)
    A, l, r = gen.cayley_automaton("Zn", generators=[1, 2], group_n=5)
    assert A.shape == (2, 2, 5, 5) and _unitary_error([A] * 6, l, r) < 1e-10
    with pytest.raises(ValueError, match="generators are required"):
        gen.cayley_automaton("Zn", group_n=5)
    with pytest.raises(ValueError, match="at least two"):
        gen.cayley_automaton("S3", generators=[(1, 0, 2)])
    with pytest.raises(ValueError, match="not an element"):
        gen.cayley_automaton("Zn", generators=[1, 9], group_n=5)


@pytest.mark.parametrize(
    "kw",
    [
        {"transitions": [[0, 1]]},  # d = 1
        {"transitions": [[0, 1], [0]]},  # ragged
        {"transitions": [[0, 5], [1, 0]]},  # out of range
        {"D": 0},
        {"D": 3, "start": 3},
        {"D": 3, "R": np.ones((2, 2))},  # not unitary
        {"D": 3, "V": [np.eye(2)] * 2},  # wrong count
        {"D": 3, "final_phases": [0.0, 1.0]},  # wrong length
    ],
)
def test_shared_basis_automaton_rejects_invalid_input(kw):
    """Malformed arguments raise ``ValueError``."""
    with pytest.raises(ValueError):
        UniformMPUGenerator().shared_basis_automaton(**kw)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_shared_basis_ring_keeps_a_longer_operator_tail_than_a_haar_basis_ring(seed):
    """Numerical property behind the notes: the weight outside a window decays less.

    Fixed seeds; the gap at radius 3 on 9 sites is at least 0.2 for these
    draws (shared basis 0.74-0.99, Haar-basis ring 0.23-0.54).
    """
    n, d, D, radius = 9, 2, 4, 3
    gen = UniformMPUGenerator(seed=seed)
    A, l, r = gen.shared_basis_automaton(D=D)
    ring = {v: [v, (v + 1) % D] for v in range(D)}
    tensors, l_h, r_h = gen.branching(d=d, bond_dim=D, n_sites=n, digraph=ring)
    U_shared = dense_unitary_from_tensors([A] * n, l, r)
    U_haar = dense_unitary_from_tensors(tensors, l_h, r_h)
    w_shared = max(_tail_weight(U_shared, n, d, 0, P, radius) for P in (PAULI_X, PAULI_Z))
    w_haar = max(_tail_weight(U_haar, n, d, 0, P, radius) for P in (PAULI_X, PAULI_Z))
    assert w_shared > w_haar + 0.15


# --- Mode II cyclic with bond_dim > d -------------------------------------------------


@pytest.mark.parametrize("cls", [MPUGenerator, UniformMPUGenerator])
@pytest.mark.parametrize("m", [3, 4, 5])
def test_cyclic_bond_dim_above_d_is_unitary(cls, m):
    """``bond_dim > d`` is accepted and the chain is unitary (verification enabled)."""
    tensors, l, r = cls(seed=2).fusion(d=2, bond_dim=m, n_sites=8, skeleton="cyclic")
    assert tensors[0].shape == (2, 2, m, m)
    assert _unitary_error(tensors, l, r) < 1e-10


@pytest.mark.parametrize("m", [3, 4])
def test_cyclic_bond_dim_above_d_commutes_with_the_charge_and_has_rank_m(m):
    """``U = R f(Q) R^dag`` commutes with ``y_1^{(x)N}``; the cut rank is ``bond_dim``."""
    n = 8
    tensors, l, r = UniformMPUGenerator(seed=3).fusion(
        d=2, bond_dim=m, n_sites=n, skeleton="cyclic"
    )
    U = dense_unitary_from_tensors(tensors, l, r)
    y1 = tensors[0][:, :, 1, 1].T
    charge = y1
    for _ in range(n - 1):
        charge = np.kron(charge, y1)
    assert np.linalg.norm(U @ charge - charge @ U) < 1e-10
    assert _operator_schmidt_rank(U, n, 2) == m


@pytest.mark.parametrize("m", [3, 4, 5])
def test_cyclic_bond_dim_above_d_auto_blocking_matches_the_charge_range(m):
    """Assumption 1 first holds on blocks of ``ceil((m-1)/(d-1))`` sites."""
    tensors, l, r = UniformMPUGenerator(seed=1).fusion(
        d=2, bond_dim=m, n_blocks=3, skeleton="cyclic", blocking="auto", max_blocking=6
    )
    assert tensors[0].shape[0] == 2 ** (m - 1)
    assert satisfies_assumption_1(tensors, l, r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # m = 3 is a short chain and warns
        shorter, *_ = UniformMPUGenerator(seed=1).fusion(
            d=2, bond_dim=m, n_blocks=3, skeleton="cyclic", blocking=m - 2
        )
    assert not satisfies_assumption_1(shorter, l, r)


def test_nested_flag_and_matrix_units_keep_their_size_constraints():
    """Only the ``cyclic`` constraint was lifted."""
    gen = UniformMPUGenerator()
    with pytest.raises(ValueError, match="nested_flag"):
        gen.fusion(d=2, bond_dim=3, n_blocks=2, skeleton="nested_flag")
    with pytest.raises(ValueError, match="matrix_units"):
        gen.fusion(d=2, bond_dim=3, n_blocks=2, skeleton="matrix_units")
