"""Regression fingerprints of the generator outputs.

Every construction (all modes, both classes, hand-picked and random options,
blocking, the named families) is sampled with fixed seeds and compared with
fingerprints stored in ``golden_outputs.json``. A fingerprint is the inner
product of the flattened tensor with a fixed pseudo-random vector, which is
insensitive to the last bits (LAPACK differs between platforms) and sensitive
to any change of the sampled ensemble, of the draw order, or of the tensor
convention. Refactors of the generator internals must leave these unchanged.

Regenerate (only for an intended change of the outputs)::

    python tests/test_golden_outputs.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mpu_generators import FAMILIES, MPUGenerator, UniformMPUGenerator, build_family
from mpu_generators.skeletons import lee_yang_skeleton

GOLDEN = Path(__file__).with_name("golden_outputs.json")
ATOL = 1e-8
_RING4 = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}

#: label -> callable(cls, seed) returning (tensors, l_vec, r_vec)
CONFIGS = {
    "branching/default": lambda c, s: c(seed=s).branching(d=2, bond_dim=2, n_sites=4),
    "branching/d3D4": lambda c, s: c(seed=s).branching(d=3, bond_dim=4, n_blocks=3),
    "branching/auto": lambda c, s: c(seed=s).branching(
        d=2, bond_dim=3, n_blocks=2, blocking="auto", max_blocking=3
    ),
    "branching/blocking2": lambda c, s: c(seed=s).branching(d=2, bond_dim=3, n_sites=6, blocking=2),
    "branching/digraph": lambda c, s: c(seed=s).branching(
        d=2, bond_dim=4, n_sites=4, digraph=_RING4
    ),
    "branching/digraph-ranks": lambda c, s: c(seed=s).branching(
        d=3, bond_dim=2, n_sites=3, digraph={0: [0, 1], 1: [1, 0]}, ranks={0: [1, 2], 1: [2, 1]}
    ),
    "branching/out_degree_random": lambda c, s: c(seed=s).branching(
        d=3, bond_dim=5, n_sites=4, out_degree="random", ranks="random"
    ),
    "branching/readout": lambda c, s: c(seed=s).branching(
        d=2, bond_dim=4, n_sites=6, readout=[0, 1, 2, 3]
    ),
    "fusion/cyclic": lambda c, s: c(seed=s).fusion(d=3, bond_dim=3, n_sites=4),
    "fusion/cyclic-M4-d3": lambda c, s: c(seed=s).fusion(
        d=3, bond_dim=4, n_sites=6, skeleton="cyclic"
    ),
    "fusion/nested_flag": lambda c, s: c(seed=s).fusion(
        d=3, bond_dim=3, n_sites=4, skeleton="nested_flag"
    ),
    "fusion/matrix_units": lambda c, s: c(seed=s).fusion(
        d=3, bond_dim=5, n_sites=3, skeleton="matrix_units"
    ),
    "fusion/group-Zn": lambda c, s: c(seed=s).fusion(
        d=3, bond_dim=3, n_sites=4, skeleton="group", group="Zn", group_n=3
    ),
    "fusion/group-S3": lambda c, s: c(seed=s).fusion(
        d=6, bond_dim=6, n_sites=3, skeleton="group", group="S3"
    ),
    "fusion/group-Z2xZ2": lambda c, s: c(seed=s).fusion(
        d=4, bond_dim=4, n_sites=3, skeleton="group", group="Zn_product", group_n=[2, 2]
    ),
    "fusion_bond_resolved/lee_yang": lambda c, s: c(seed=s).fusion_bond_resolved(
        lee_yang_skeleton(), n_sites=3
    ),
    "grafted/default": lambda c, s: c(seed=s).grafted(d=3, bond_dim=4, n_sites=4),
    "shared_basis_chain/ring": lambda c, s: c(seed=s).shared_basis_chain(
        n_sites=4, d=2, D=3, transitions="ring"
    ),
    "shared_basis_chain/ring-auto": lambda c, s: c(seed=s).shared_basis_chain(
        n_blocks=2, d=2, D=4, transitions="ring", blocking="auto", max_blocking=3
    ),
    "cayley_chain/S3": lambda c, s: c(seed=s).cayley_chain(n_sites=3),
    "multi_start/ring": lambda c, s: c(seed=s).multi_start(n_sites=4, D=2, transitions="ring"),
    "multi_start/no-edge-phases": lambda c, s: c(seed=s).multi_start(
        n_sites=4, D=2, transitions="ring", edge_phases=None
    ),
    **{
        f"family/{name}": (lambda c, s, name=name: build_family(name, seed=s))
        for name in sorted(FAMILIES)
    },
}


def _vector(size: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.standard_normal(size) + 1j * rng.standard_normal(size)


def fingerprint(result) -> list[list[float]]:
    """Return ``[[re, im], ...]`` fingerprints of every tensor and of ``l_vec``, ``r_vec``."""
    tensors, l_vec, r_vec = result
    arrays = [*tensors, l_vec, r_vec]
    return [
        [float(z.real), float(z.imag)]
        for z in (np.vdot(_vector(a.size), np.asarray(a).ravel()) for a in arrays)
    ]


def _run(label: str, cls, seed: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CONFIGS[label](cls, seed)


def _key(label: str, cls, seed: int) -> str:
    return f"{label}|{cls.__name__}|{seed}"


def _cases():
    for label in CONFIGS:
        if label.startswith("family/"):
            yield label, MPUGenerator, 0  # build_family fixes the class itself
            yield label, MPUGenerator, 3
            continue
        for cls in (MPUGenerator, UniformMPUGenerator):
            for seed in (0, 3):
                yield label, cls, seed


@pytest.mark.parametrize("label,cls,seed", list(_cases()), ids=lambda v: getattr(v, "__name__", v))
def test_outputs_match_the_stored_fingerprints(label, cls, seed):
    """The sampled tensors and boundaries are unchanged."""
    golden = json.loads(GOLDEN.read_text())
    key = _key(label, cls, seed)
    assert key in golden, f"no stored fingerprint for {key}; regenerate if the case is new"
    got = np.array(fingerprint(_run(label, cls, seed)))
    ref = np.array(golden[key])
    assert got.shape == ref.shape, f"{key}: number of tensors changed"
    np.testing.assert_allclose(got, ref, atol=ATOL, err_msg=key)


if __name__ == "__main__":
    out = {
        _key(label, cls, seed): fingerprint(_run(label, cls, seed)) for label, cls, seed in _cases()
    }
    GOLDEN.write_text(json.dumps(out, indent=0))
    print(f"wrote {len(out)} fingerprints to {GOLDEN}")
