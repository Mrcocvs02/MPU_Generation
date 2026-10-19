"""Mode I ensemble options: hand-picked / random digraphs and rank partitions, per-N draws.

Every option must stay inside the certified class (exact unitarity for every
N) and leave the default ensemble bit-identical.
"""

import numpy as np
import pytest
from TreeMPU.checks import dense_unitary_from_tensors

from mpu_generators import MPUGenerator, UniformMPUGenerator
from mpu_generators.base import _random_composition


def _unitary_error(t, l, r):
    U = dense_unitary_from_tensors(t, l, r)
    return np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))


def _block_ranks(A, v):
    """Ranks of the populated blocks of row v, in target order."""
    D = A.shape[2]
    return {
        u: int(np.linalg.matrix_rank(A[:, :, v, u], tol=1e-9))
        for u in range(D)
        if np.any(A[:, :, v, u])
    }


DIGRAPH = {0: [0, 2], 1: [1], 2: [0, 1, 2]}


@pytest.mark.parametrize("cls", [MPUGenerator, UniformMPUGenerator])
@pytest.mark.parametrize(
    "kw",
    [
        {"digraph": DIGRAPH},
        {"digraph": DIGRAPH, "ranks": {0: [1, 2], 1: [3], 2: [1, 1, 1]}},
        {"digraph": DIGRAPH, "ranks": "random"},
        {"out_degree": "random"},
        {"out_degree": 2, "ranks": "random"},
        {"out_degree": 1},
    ],
)
def test_mode1_options_stay_unitary(cls, kw):
    for N in (2, 3, 4):
        t, l, r = cls(seed=5).branching(d=3, bond_dim=3, n_sites=N, **kw)
        assert _unitary_error(t, l, r) < 1e-10


def test_handpicked_digraph_and_ranks_are_used():
    ranks = {0: [1, 2], 1: [3], 2: [1, 1, 1]}
    t, _, _ = UniformMPUGenerator(seed=0).branching(
        d=3, bond_dim=3, n_sites=2, digraph=DIGRAPH, ranks=ranks
    )
    A = t[0]
    for v, targets in DIGRAPH.items():
        got = _block_ranks(A, v)
        assert sorted(got) == sorted(targets)
        assert [got[u] for u in targets] == ranks[v]


def test_random_ranks_cover_unbalanced_partitions():
    seen = set()
    for seed in range(40):
        t, _, _ = UniformMPUGenerator(seed=seed).branching(
            d=4, bond_dim=2, n_sites=2, digraph={0: [0, 1], 1: [0, 1]}, ranks="random"
        )
        seen.add(tuple(sorted(_block_ranks(t[0], 0).values())))
    assert seen == {(1, 3), (2, 2)}


def test_random_composition_is_uniform():
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(6000):
        c = tuple(_random_composition(4, 2, rng))
        counts[c] = counts.get(c, 0) + 1
    assert set(counts) == {(1, 3), (2, 2), (3, 1)}
    assert all(abs(v / 6000 - 1 / 3) < 0.03 for v in counts.values())


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"digraph": {0: [0], 1: []}}, "out-degree"),
        ({"digraph": {0: [0, 0], 1: [1]}}, "distinct"),
        ({"digraph": {0: [0, 5], 1: [1]}}, "distinct"),
        ({"digraph": {0: [0]}}, "vertices"),
        ({"digraph": {0: [0, 1], 1: [1]}, "ranks": {0: [2, 0], 1: [2]}}, "positive"),
        ({"ranks": {0: [1, 1], 1: [2]}}, "explicit digraph"),
    ],
)
def test_invalid_mode1_options_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        MPUGenerator(seed=0).branching(d=2, bond_dim=2, n_sites=2, **kw)


def test_default_ensemble_is_unchanged_by_the_new_options():
    a = UniformMPUGenerator(seed=3).branching(d=2, bond_dim=2, n_sites=3)
    b = UniformMPUGenerator(seed=3).branching(d=2, bond_dim=2, n_sites=3, ranks="balanced")
    for x, y in zip(a[0], b[0], strict=True):
        assert np.array_equal(x, y)


def test_coherent_in_n_switch():
    coherent = [
        UniformMPUGenerator(seed=2).branching(d=2, bond_dim=2, n_sites=N)[0][0] for N in (2, 3)
    ]
    assert np.array_equal(coherent[0], coherent[1])
    per_n = [
        UniformMPUGenerator(seed=2, coherent_in_n=False).branching(d=2, bond_dim=2, n_sites=N)[0][0]
        for N in (2, 3)
    ]
    assert not np.allclose(per_n[0], per_n[1])
    # still reproducible for a fixed (seed, N)
    again = UniformMPUGenerator(seed=2, coherent_in_n=False).branching(d=2, bond_dim=2, n_sites=3)[
        0
    ][0]
    assert np.array_equal(per_n[1], again)
    t, l, r = MPUGenerator(seed=2, coherent_in_n=False).fusion(d=2, bond_dim=2, n_sites=3)
    assert _unitary_error(t, l, r) < 1e-10
