"""Site-dependent shared-basis chains: ``shared_basis_chain`` and ``cayley_chain``.

At every site all vertices read the input in one basis; nothing couples the
bases (or the vertex unitaries, or the bond dimensions) of different sites.
The tests check exact unitarity for site-dependent data, the action on the
product reading basis written out independently of the tensors, the reduction
to the translation-invariant automata, the draw order, input validation, and
the operator tail that separates permutations from coalescing maps.
"""

import numpy as np
import pytest
from TreeMPU.checks import dense_unitary_from_tensors

from mpu_generators import MPUGenerator, UniformMPUGenerator
from mpu_generators.base import _haar_unitary


def _unitary_error(tensors, l, r):
    """Return the Frobenius norm of ``U^dag U - 1`` for the dense chain."""
    U = dense_unitary_from_tensors(tensors, l, r)
    return np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))


def _tail_weight(U, n, d, site, P, radius):
    """Fraction of ``U P_site U^dag`` not supported in ``[site-radius, site+radius]``."""
    op = U @ np.kron(np.kron(np.eye(d ** (n - site - 1)), P), np.eye(d**site)) @ U.conj().T
    keep = [n - 1 - k for k in range(max(0, site - radius), min(n - 1, site + radius) + 1)]
    out = [a for a in range(n) if a not in keep]
    t = op.reshape([d] * (2 * n)).transpose(
        keep + out + [n + a for a in keep] + [n + a for a in out]
    )
    t = t.reshape(d ** len(keep), d ** len(out), d ** len(keep), d ** len(out))
    reduced = np.einsum("aobo->ab", t) / d ** len(out)
    return 1 - np.linalg.norm(reduced) ** 2 * d ** len(out) / np.linalg.norm(op) ** 2


PAULI_X = np.array([[0, 1.0], [1.0, 0]])
PAULI_Z = np.diag([1.0, -1.0])


# --- unitarity ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "kw",
    [
        {"n_sites": 4, "d": 2, "D": 3},
        {"n_sites": 5, "d": 2, "D": 4, "transitions": "random"},
        {"n_sites": 4, "d": 3, "D": [3, 4, 2, 5, 3]},
        {"n_sites": 5, "d": 2, "D": [1, 3, 2, 6, 2, 4], "final_phases": "random"},
        {"n_sites": 3, "d": 3, "D": [2, 5, 5, 5], "start": 1},
        {"n_sites": 6, "d": 2, "D": 3, "transitions": "ring", "blocking": 2},
        {"n_blocks": 2, "d": 2, "D": 5, "blocking": 3},
    ],
)
def test_shared_basis_chain_is_unitary(kw, seed):
    """Random maps, variable bond dimensions and blocking all give a unitary chain."""
    tensors, l, r = MPUGenerator(seed=seed).shared_basis_chain(**kw)
    assert _unitary_error(tensors, l, r) < 1e-10


def test_shared_basis_chain_differs_from_site_to_site():
    """Each site gets its own reading basis, vertex unitaries and maps."""
    tensors, _, _ = MPUGenerator(seed=0).shared_basis_chain(n_sites=3, d=2, D=4)
    assert not np.allclose(tensors[0], tensors[1])
    assert not np.allclose(tensors[1], tensors[2])


def test_shared_basis_chain_acts_as_the_product_reading_basis_formula():
    """``U |r_x1 .. r_xN> = e^{i phi_{v_N}} (x) V_{v_{k-1}} r_{k, x_k}``, from the definition."""
    rng = np.random.default_rng(5)
    n, d, Ds = 4, 3, [2, 4, 3, 5, 3]
    R = [_haar_unitary(d, rng) for _ in range(n)]
    V = [[_haar_unitary(d, rng) for _ in range(Ds[k])] for k in range(n)]
    sigma = [
        [[int(u) for u in rng.integers(0, Ds[k + 1], size=Ds[k])] for _ in range(d)]
        for k in range(n)
    ]
    phases = rng.uniform(0, 2 * np.pi, size=Ds[-1])
    start = 1
    tensors, l, r = MPUGenerator().shared_basis_chain(
        n_sites=n, d=d, D=Ds, transitions=sigma, R=R, V=V, final_phases=phases, start=start
    )
    U = dense_unitary_from_tensors(tensors, l, r)  # U[out, in], site 0 least significant
    for x in [(0, 0, 0, 0), (2, 1, 0, 2), (1, 2, 2, 1)]:
        v, out_sites, in_sites = start, [], []
        for k in range(n):
            in_sites.append(R[k][:, x[k]])
            out_sites.append(V[k][v] @ R[k][:, x[k]])
            v = sigma[k][x[k]][v]
        psi_in = np.array([1.0 + 0j])
        psi_out = np.array([1.0 + 0j])
        for k in range(n):  # little endian: site k is the k-th factor from the right
            psi_in = np.kron(in_sites[k], psi_in)
            psi_out = np.kron(out_sites[k], psi_out)
        assert np.linalg.norm(U @ psi_in - np.exp(1j * phases[v]) * psi_out) < 1e-10


# --- reduction to the translation-invariant automata ----------------------------------


@pytest.mark.parametrize("n", [3, 5])
def test_uniform_chain_repeats_one_tensor_and_matches_the_single_tensor_method(n):
    """``UniformMPUGenerator.shared_basis_chain`` reuses the tensor of the single-tensor method."""
    rng = np.random.default_rng(2)
    d, D = 2, 4
    R = _haar_unitary(d, rng)
    V = [_haar_unitary(d, rng) for _ in range(D)]
    phases = rng.uniform(0, 2 * np.pi, size=D)
    gen = UniformMPUGenerator(seed=0)
    A, l, r = gen.shared_basis_automaton(D=D, d=d, R=R, V=V, final_phases=phases)
    tensors, l2, r2 = gen.shared_basis_chain(
        n_sites=n, d=d, D=D, transitions="ring", R=R, V=V, final_phases=phases
    )
    assert all(np.allclose(t, A) for t in tensors)
    assert np.allclose(l, l2) and np.allclose(r, r2)


def test_site_dependent_chain_with_equal_data_equals_the_uniform_chain():
    """Giving the same ``R``, ``V``, maps at every site removes the site dependence."""
    rng = np.random.default_rng(3)
    d, D, n = 2, 3, 4
    R = _haar_unitary(d, rng)
    V = [_haar_unitary(d, rng) for _ in range(D)]
    kw = {"n_sites": n, "d": d, "D": D, "transitions": "ring", "R": R, "V": V}
    site_dep, _, _ = MPUGenerator(seed=1).shared_basis_chain(**kw)
    uniform, _, _ = UniformMPUGenerator(seed=7).shared_basis_chain(**kw)
    for a, b in zip(site_dep, uniform):
        assert np.allclose(a, b)


def test_per_site_data_matches_repeated_single_data():
    """``R[k]``, ``V[k][v]`` and ``transitions[k][a][v]`` repeated equal the single forms."""
    rng = np.random.default_rng(4)
    d, D, n = 2, 3, 3
    R = _haar_unitary(d, rng)
    V = [_haar_unitary(d, rng) for _ in range(D)]
    t = [[(v + a) % D for v in range(D)] for a in range(d)]
    gen = MPUGenerator()
    single, _, _ = gen.shared_basis_chain(n_sites=n, d=d, D=D, transitions=t, R=R, V=V)
    per_site, _, _ = gen.shared_basis_chain(
        n_sites=n, d=d, D=D, transitions=[t] * n, R=[R] * n, V=[V] * n
    )
    for a, b in zip(single, per_site):
        assert np.allclose(a, b)


def test_chain_of_n_plus_one_sites_extends_the_chain_of_n_sites_at_the_left_end():
    """For a constant ``D`` the draws follow the construction order common to all modes."""
    gen = MPUGenerator(seed=3)
    short, *_ = gen.shared_basis_chain(n_sites=3, d=2, D=3)
    long, *_ = gen.shared_basis_chain(n_sites=4, d=2, D=3)
    for a, b in zip(short, long[1:]):
        assert np.allclose(a, b)


@pytest.mark.parametrize("cls", [MPUGenerator, UniformMPUGenerator])
def test_chain_is_reproducible(cls):
    """Same seed, same chain."""
    a, *_ = cls(seed=11).shared_basis_chain(n_sites=3, d=2, D=3)
    b, *_ = cls(seed=11).shared_basis_chain(n_sites=3, d=2, D=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --- Cayley chains ---------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MPUGenerator, UniformMPUGenerator])
@pytest.mark.parametrize("seed", [0, 1])
def test_cayley_chain_s3_is_unitary(cls, seed):
    """S3 with a transposition and a 3-cycle: ``d = 2``, ``D = 6``."""
    tensors, l, r = cls(seed=seed).cayley_chain(n_sites=4)
    assert tensors[0].shape == (2, 2, 6, 6)
    assert _unitary_error(tensors, l, r) < 1e-10


def test_cayley_chain_matches_the_single_tensor_method_when_the_data_agree():
    """Same ``R`` and ``V`` at every site: the chain repeats ``cayley_automaton``."""
    rng = np.random.default_rng(6)
    R = _haar_unitary(2, rng)
    V = [_haar_unitary(2, rng) for _ in range(6)]
    tensors, l2, r2 = UniformMPUGenerator(seed=0).cayley_chain(n_sites=3, R=R, V=V)
    ref, l3, r3 = UniformMPUGenerator(seed=0).cayley_automaton(R=R, V=V)
    assert all(np.allclose(t, ref) for t in tensors)
    assert np.allclose(l2, l3) and np.allclose(r2, r3)


def test_cayley_chain_site_dependent_bases_with_zn():
    """Z4 with two generators, different bases and unitaries at each site."""
    tensors, l, r = MPUGenerator(seed=2).cayley_chain(
        n_sites=3, group="Zn", group_n=4, generators=[1, 3], final_phases="random"
    )
    assert tensors[0].shape == (2, 2, 4, 4)
    assert not np.allclose(tensors[0], tensors[1])
    assert _unitary_error(tensors, l, r) < 1e-10


# --- validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"d": 2, "D": 3},  # neither n_sites nor n_blocks
        {"n_sites": 4, "n_blocks": 2, "d": 2, "D": 3},
        {"n_sites": 4, "d": 1, "D": 3},
        {"n_sites": 4, "d": 2, "D": 0},
        {"n_sites": 4, "d": 2, "D": [3, 3, 3]},  # wrong length
        {"n_sites": 4, "d": 2, "D": [3, 4, 3, 3, 3], "transitions": "permutation"},
        {"n_sites": 4, "d": 2, "D": [3, 4, 3, 3, 3], "transitions": "ring"},
        {"n_sites": 4, "d": 2, "D": 3, "transitions": "nonsense"},
        {"n_sites": 4, "d": 2, "D": 3, "start": 3},
        {"n_sites": 4, "d": 2, "D": 3, "R": np.ones((2, 2))},
        {"n_sites": 4, "d": 2, "D": 3, "V": [np.eye(2)] * 2},  # too few
        {"n_sites": 4, "d": 2, "D": 3, "R": [np.eye(2)] * 3},  # wrong number of sites
        {"n_sites": 4, "d": 2, "D": 3, "final_phases": [0.0, 1.0]},
        {"n_sites": 4, "d": 2, "D": 3, "final_phases": "uniform"},
        {"n_sites": 4, "d": 2, "D": 3, "transitions": [[[0, 1, 5]] * 2] * 4},  # entry out of range
        {"n_sites": 4, "d": 2, "D": 3, "transitions": [[[0, 1]] * 2] * 4},  # wrong length
        {"n_sites": 5, "d": 2, "D": 3, "blocking": 2},  # not divisible
        {"n_sites": 2, "d": 2, "D": 3, "blocking": 2},  # a single tensor
    ],
)
def test_shared_basis_chain_rejects_invalid_input(kw):
    """Malformed arguments raise ``ValueError``."""
    with pytest.raises(ValueError):
        MPUGenerator().shared_basis_chain(**kw)


def test_uniform_generator_rejects_per_site_data_and_variable_bonds():
    """A translation-invariant generator cannot take site-dependent data."""
    gen = UniformMPUGenerator()
    with pytest.raises(ValueError, match="site-dependent"):
        gen.shared_basis_chain(n_sites=3, d=2, D=3, R=[np.eye(2)] * 3)
    with pytest.raises(ValueError, match="integer"):
        gen.shared_basis_chain(n_sites=3, d=2, D=[3, 3, 3, 3])


@pytest.mark.parametrize(
    "kw",
    [
        {"group": "Zn", "group_n": 4},  # generators required
        {"generators": [(0, 1, 2)]},  # one generator
        {"generators": [(0, 1, 2), (5, 5, 5)]},  # not an element
    ],
)
def test_cayley_chain_rejects_invalid_input(kw):
    """Missing, too few or non-element generators raise ``ValueError``."""
    with pytest.raises(ValueError):
        MPUGenerator().cayley_chain(n_sites=3, **kw)


# --- operator tail ----------------------------------------------------------------------


def _site0_tail(tensors, l, r, n, d, radius):
    """Worst of Pauli X and Z of the weight of ``U P_0 U^dag`` outside the window."""
    U = dense_unitary_from_tensors(tensors, l, r)
    return max(_tail_weight(U, n, d, 0, P, radius) for P in (PAULI_X, PAULI_Z))


def test_permutation_tails_do_not_change_with_the_radius_and_coalescing_maps_lose_weight():
    """Permutation maps: the weight outside the window is the same at every radius.

    Fixed seeds 0-5, 8 sites, ``D = 4``, bases and vertex unitaries drawn
    independently at every site. The size of the weight depends on the draw
    (0.55-0.99 here), so only its dependence on the radius is asserted: for the
    ring and for random permutations it changes by less than 0.02 between
    radius 1 and 3 (observed at most 0.005); for random maps, which coalesce,
    it never increases and drops on average by about 0.16 (up to 0.43).
    """
    n, d, D = 8, 2, 4
    drop = {}
    for mode in ("ring", "permutation", "random"):
        drop[mode] = []
        for s in range(6):
            tensors, l, r = MPUGenerator(seed=s).shared_basis_chain(
                n_sites=n, d=d, D=D, transitions=mode
            )
            w1 = _site0_tail(tensors, l, r, n, d, 1)
            w3 = _site0_tail(tensors, l, r, n, d, 3)
            if mode != "random":
                assert w3 > 0.3
            drop[mode].append(w1 - w3)
    assert max(abs(x) for x in drop["ring"]) < 0.02
    assert max(abs(x) for x in drop["permutation"]) < 0.02
    assert min(drop["random"]) > -1e-9
    assert np.mean(drop["random"]) > 0.08


@pytest.mark.parametrize("collide", [True, False])
def test_site0_tail_vanishes_iff_both_letters_reach_the_same_vertex_from_the_start(collide):
    """Permutations: the tail of a site-0 operator is zero when sigma_0(start) = sigma_1(start).

    Otherwise the vertex of every later site depends on the letter at site 0
    and the weight beyond the window is large (more than 0.5 is asserted).
    """
    n, d, D, radius = 8, 2, 4, 3
    rng = np.random.default_rng(0)
    transitions = []
    for k in range(n):
        a, b = list(rng.permutation(D)), list(rng.permutation(D))
        if k == 0 and (a[0] == b[0]) != collide:
            j = b.index(a[0]) if collide else 1
            b[0], b[j] = b[j], b[0]
        transitions.append([a, b])
    tensors, l, r = MPUGenerator(seed=0).shared_basis_chain(
        n_sites=n, d=d, D=D, transitions=transitions
    )
    weight = _site0_tail(tensors, l, r, n, d, radius)
    if collide:
        assert weight < 1e-9
    else:
        assert weight > 0.5
