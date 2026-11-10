"""Mode I with several start vertices (``multi_start``)."""

import numpy as np
import pytest
from TreeMPU.checks import dense_unitary_from_tensors as dense

from mpu_generators import MPUGenerator, UniformMPUGenerator, mixing
from mpu_generators.base import _check_row_resolution
from mpu_generators.skeletons import MultiStartSkeleton


def _error(chain):
    U = dense(*chain)
    return np.linalg.norm(U.conj().T @ U - np.eye(len(U))) / np.sqrt(len(U))


def _first_cut_rank(U, n, d, tol=1e-9):
    T = U.reshape([d] * (2 * n))
    rest = [a for a in range(2 * n) if a not in (n - 1, 2 * n - 1)]
    M = T.transpose([n - 1, 2 * n - 1, *rest]).reshape(d * d, -1)
    return int((np.linalg.svd(M, compute_uv=False) > tol).sum())


def _blocks(A, size):
    n = A.shape[2] // size
    return [A[:, :, b * size : (b + 1) * size, b * size : (b + 1) * size] for b in range(n)]


@pytest.mark.parametrize("cls", [MPUGenerator, UniformMPUGenerator])
@pytest.mark.parametrize(
    "options",
    [
        {"D": 1},
        {"D": 3, "transitions": "ring"},
        {"D": 3, "transitions": "random"},
        {"d": 3, "n_sectors": 3, "D": 2, "transitions": "ring"},
        {"d": 3, "n_sectors": 2, "D": 2, "transitions": "ring"},
        {"D": 2, "edge_phases": None},
        {"d": 3, "n_sectors": 2, "D": 2, "charges": [1, 1, 0]},
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_unitary_for_any_data(cls, options, seed):
    """The chain is unitary for random bases, unitaries, maps and phases."""
    assert _error(cls(seed=seed).multi_start(n_sites=4, **options)) < 1e-12


def test_unitary_with_varying_bond_dimensions():
    """Site-dependent bond dimensions are allowed, as in the automaton."""
    t, l_vec, r_vec = MPUGenerator(seed=2).multi_start(
        n_sites=4, D=[2, 3, 2, 3, 2], transitions="random"
    )
    assert [A.shape[2:] for A in t] == [(8, 12), (12, 8), (8, 12), (12, 8)]
    assert _error((t, l_vec, r_vec)) < 1e-12


def test_uniform_chain_keeps_tensors_and_boundaries_for_every_length():
    """One tensor and fixed boundaries give a unitary at every length."""
    t, l_vec, r_vec = UniformMPUGenerator(seed=1).multi_start(n_sites=2, D=2)
    for n in range(1, 8):
        assert _error(([t[0]] * n, l_vec, r_vec)) < 1e-12


def test_every_block_is_a_mode_I_chain():
    """Each copy satisfies (RR) and, with its own start and end, is unitary."""
    t, l_vec, r_vec = MPUGenerator(seed=4).multi_start(n_sites=4, D=3, transitions="ring")
    start = np.zeros(3, dtype=complex)
    start[0] = 1
    for b in range(4):
        sites = [_blocks(A, 3)[b] for A in t]
        for A in sites:
            _check_row_resolution(A)
        assert _error((sites, start, r_vec[3 * b : 3 * b + 3])) < 1e-12
    assert np.count_nonzero(l_vec) == 4


def test_one_sector_without_phases_is_the_automaton():
    """``n_sectors=1`` and no edge phases reproduce ``shared_basis_chain``."""
    options = {"n_sites": 4, "D": 3, "transitions": "ring"}
    a = UniformMPUGenerator(seed=3).shared_basis_chain(**options)
    b = UniformMPUGenerator(seed=3).multi_start(
        n_sectors=1, sector_phases=[0.0], edge_phases=None, **options
    )
    np.testing.assert_allclose(dense(*a), dense(*b), atol=1e-13)


def test_no_edge_phases_is_a_mode_II_convolution():
    """With ``D=1`` and no edge phases the sum is two phases on the parity sectors."""
    n, phi = 5, [0.3, 1.9]
    t, l_vec, r_vec = UniformMPUGenerator(seed=0).multi_start(
        n_sites=n, D=1, V=[np.eye(2)], R=np.eye(2), sector_phases=phi, edge_phases=None
    )
    assert t[0].shape[2] == 2
    parity = np.array([x.bit_count() % 2 for x in range(2**n)])
    np.testing.assert_allclose(
        dense(t, l_vec, r_vec), np.diag(np.exp(1j * np.array(phi)[parity])), atol=1e-13
    )


def test_guess_selects_the_phase_of_the_true_charge():
    """On a basis state the amplitude is the phase of the guess equal to the true charge."""
    n, theta = 6, 1.0
    t, l_vec, r_vec = UniformMPUGenerator(seed=0).multi_start(
        n_sites=n,
        D=1,
        V=[np.eye(2)],
        R=np.eye(2),
        sector_phases=[0.0, 0.0],
        edge_phases=np.array([[[0.0, 0.0]], [[0.0, theta]]]),
    )
    ones = np.array([x.bit_count() for x in range(2**n)])
    expected = np.exp(1j * np.where(ones % 2 == 0, 0.0, theta * ones))
    np.testing.assert_allclose(np.diag(dense(t, l_vec, r_vec)), expected, atol=1e-13)


def test_a_single_start_needs_a_growing_number_of_vertices():
    """The rest operators after a prefix are j + 1 classes at level j, at fixed bond 4."""
    n, theta = 10, 1.0
    ones = np.array([x.bit_count() for x in range(2**n)])
    phases = np.where(ones % 2 == 0, 0.0, theta * ones)
    for j in range(1, n - 1):
        table = phases.reshape(2**j, -1)
        classes = {tuple(np.round(np.exp(1j * (row - row[0])), 7)) for row in table}
        assert len(classes) == j + 1


def test_first_cut_rank_stays_at_d_until_stacked():
    """All copies read in one basis; another reading basis takes the rank above d."""
    n = 5
    chain = UniformMPUGenerator(seed=1).multi_start(n_sites=n, D=2, transitions="ring")
    assert _first_cut_rank(dense(*chain), n, 2) <= 2
    front = UniformMPUGenerator(seed=2).branching(d=2, bond_dim=2, n_sites=n)
    stacked = mixing.stack(front, chain)
    assert _error(stacked) < 1e-12
    assert _first_cut_rank(dense(*stacked), n, 2) > 2


@pytest.mark.parametrize(
    "weights",
    [(0.5, 0.5, 0.5, 0.5), (0.6, 0.4, 0.5, -0.5), (0.55, 0.55, 0.45, -0.45)],
)
def test_start_weights_are_not_free(weights):
    """Other start weights break unitarity."""
    t, _l, r_vec = UniformMPUGenerator(seed=0).multi_start(n_sites=4, D=1, sector_phases=[0.0, 0.0])
    assert _error((t, np.array(weights, dtype=complex), r_vec)) > 0.05


def _ranks(chain, n, d=2, tol=1e-9):
    T = dense(*chain).reshape([d] * (2 * n))
    out = []
    for k in range(1, n):
        left = [n - 1 - j for j in range(k)] + [2 * n - 1 - j for j in range(k)]
        right = [a for a in range(2 * n) if a not in left]
        M = T.transpose(left + right).reshape(d ** (2 * k), -1)
        out.append(int((np.linalg.svd(M, compute_uv=False) > tol).sum()))
    return out


def test_an_even_ring_makes_the_fourier_blocks_redundant():
    """The end vertex knows the parity: rank 2D on an even ring, up to 4D on an odd one."""
    n = 9
    even = UniformMPUGenerator(seed=1).multi_start(n_sites=n, D=2, transitions="ring")
    odd = UniformMPUGenerator(seed=1).multi_start(n_sites=n, D=3, transitions="ring")
    assert max(_ranks(even, n)) == 4
    assert max(_ranks(odd, n)) == 12


def test_more_sectors_than_letters():
    """Unitarity does not need ``n_sectors <= d``."""
    chain = UniformMPUGenerator(seed=0).multi_start(n_sites=4, n_sectors=3, D=1)
    assert chain[0][0].shape[2] == 9
    assert _error(chain) < 1e-12


def test_verify_false_gives_the_same_tensors():
    """The checks draw nothing."""
    a = MPUGenerator(seed=5).multi_start(n_sites=4, D=2)
    b = MPUGenerator(seed=5, verify=False).multi_start(n_sites=4, D=2)
    for x, y in zip(a[0], b[0], strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"n_sectors": 0}, "n_sectors"),
        ({"charges": [0]}, "charges"),
        ({"charges": [0.5, 1.5]}, "charges"),
        ({"sector_phases": "zero"}, "sector_phases"),
        ({"sector_phases": [0.0]}, "sector_phases"),
        ({"edge_phases": "zero"}, "edge_phases"),
        ({"edge_phases": np.zeros((2, 2, 2))}, "edge_phases"),
    ],
)
def test_invalid_data_is_rejected(options, match):
    """Wrong sizes and values raise ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        MPUGenerator(seed=0).multi_start(n_sites=3, D=4, **options)


def test_skeleton_can_be_sampled_directly():
    """The skeleton goes through the common pipeline."""
    chain = MPUGenerator(seed=0).sample(MultiStartSkeleton(2, D=2), n_sites=3)
    assert chain[0][0].shape == (2, 2, 8, 8)
    assert _error(chain) < 1e-12
