# mpu_generators/mixing.py
r"""
Ways to combine two independently-certified MPU chains into one
([PDF-B] Part IV). Operates on ``(tensors, l_vec, r_vec)`` triples in this
package's own convention, ``tensors`` a list (one tensor per site, use
``[A]*N`` for a uniform single-tensor result from ``uniform.py``).

- :func:`stack`: operator composition, bond dim ``D1*D2``, no extra site --
  trivially unitary (product of two unitaries), no certification needed
  beyond the inputs' own.
- :func:`nest`: a thin alias for
  :meth:`mpu_generators.uniform.UniformMPUGenerator.invariant_subspace` --
  [PDF-B] frames it as a distinct "mixing mode" even though it is the same
  construction as the rank-$k$ invariant subspace mechanism.
- :func:`capped_sum`: [PDF-B] §4.4, one extra site whose row is itself an
  (RR) pair $\Pi_1,\Pi_2$ (reusing
  :func:`mpu_generators.base._row_resolution_blocks` -- this is *exactly*
  Mode I's own row-resolution recipe, applied to two whole sub-chains
  instead of single branches, per the source doc's own framing), giving
  $U_{N+1}=\Pi_1\otimes U_N^{(1)}+\Pi_2\otimes U_N^{(2)}$ -- provably
  unitary for *any* two independently-certified inputs
  ($\Pi_1^\dagger\Pi_1+\Pi_2^\dagger\Pi_2=1$, $\Pi_1^\dagger\Pi_2=0$ makes
  the cross terms vanish exactly, mirroring Theorem 1's own proof).

**Deliberately not implemented**: naive summing without the extra site
($l=l_1+l_2$, no $\Pi$) -- [PDF-B] §4.3 shows this fails in general (no
reason for $U^{(1)\dagger}U^{(2)}$ cross terms to cancel); a regression test
demonstrating the failure lives in ``tests/``, not a function here.
"""
from __future__ import annotations

import numpy as np

from .base import _row_resolution_blocks


def stack(mpu1, mpu2, seed: int | None = None):
    r"""
    Operator composition ([PDF-B] §4.1): $U_N := U_N^{(1)} \cdot U_N^{(2)}$,
    bond dimension $D_1 D_2$, no extra site. Requires equal physical
    dimension $d$ and equal chain length $N$ for both inputs. ``seed`` is
    accepted for interface symmetry with the other mixing functions but is
    unused (this construction has no randomness of its own).
    """
    tensors1, l1, r1 = mpu1
    tensors2, l2, r2 = mpu2
    if len(tensors1) != len(tensors2):
        raise ValueError("stack requires equal-length chains.")
    d = tensors1[0].shape[0]
    if tensors2[0].shape[0] != d:
        raise ValueError(f"stack requires equal physical dimension d (got {d} and {tensors2[0].shape[0]}).")

    tensors = []
    for A1, A2 in zip(tensors1, tensors2):
        # Compose as operators: apply mpu2 then mpu1 (A1's p_in meets A2's
        # p_out); bond dims combine by Kronecker product.
        A = np.einsum("kjab,ikcd->ijacbd", A1, A2)
        s = A.shape
        A = A.reshape(s[0], s[1], s[2] * s[3], s[4] * s[5])
        tensors.append(A)
    l = np.kron(l1, l2).astype(complex)
    r = np.kron(r1, r2).astype(complex)
    return tensors, l, r


def nest(sub_A, sub_l, sub_r, D_extra: int = 2, seed: int | None = None, generator=None):
    r"""
    Thin alias for
    :meth:`mpu_generators.uniform.UniformMPUGenerator.invariant_subspace`,
    exposed under [PDF-B]'s own "mixing" name -- same construction, not a
    separate implementation. ``generator`` may be an existing
    ``UniformMPUGenerator`` instance (its ``seed``/``verify`` are then
    reused); a fresh one is created otherwise.
    """
    from .uniform import UniformMPUGenerator

    gen = generator if generator is not None else UniformMPUGenerator(seed=seed or 0)
    return gen.invariant_subspace(sub_A, sub_l, sub_r, D_extra=D_extra, seed=seed)


def capped_sum(mpu1, mpu2, seed: int | None = None):
    r"""
    Capped sum ([PDF-B] §4.4): one extra site prepended, carrying an (RR)
    pair $\Pi_1,\Pi_2$ (built via
    :func:`mpu_generators.base._row_resolution_blocks` on 2 targets --
    literally Mode I's own row-resolution recipe), giving
    $U_{N+1}=\Pi_1\otimes U_N^{(1)}+\Pi_2\otimes U_N^{(2)}$, unitary for
    *any* two independently-certified inputs. Requires equal physical
    dimension $d$ (this implementation's scope choice -- see module
    docstring) and equal chain length $N$ for both inputs; bond dimensions
    may differ freely.
    """
    tensors1, l1, r1 = mpu1
    tensors2, l2, r2 = mpu2
    if len(tensors1) != len(tensors2):
        raise ValueError("capped_sum requires equal-length chains.")
    d = tensors1[0].shape[0]
    if tensors2[0].shape[0] != d:
        raise ValueError(
            f"capped_sum (this implementation) requires equal physical "
            f"dimension d for both inputs (got {d} and {tensors2[0].shape[0]})."
        )
    D1, D2 = len(l1), len(l2)
    rng = np.random.default_rng(seed if seed is not None else 0)

    blocks = _row_resolution_blocks(d, [0, 1], rng)
    Pi1, Pi2 = blocks[0], blocks[1]

    D_new = D1 + D2
    A_readout = np.zeros((d, d, 1, D_new), dtype=complex)
    A_readout[:, :, 0, :D1] = Pi1[:, :, None] * l1[None, None, :]
    A_readout[:, :, 0, D1:] = Pi2[:, :, None] * l2[None, None, :]

    combined = [A_readout]
    for A1, A2 in zip(tensors1, tensors2):
        Da1, Db1 = A1.shape[2], A1.shape[3]
        Da2, Db2 = A2.shape[2], A2.shape[3]
        A = np.zeros((d, d, Da1 + Da2, Db1 + Db2), dtype=complex)
        A[:, :, :Da1, :Db1] = A1
        A[:, :, Da1:, Db1:] = A2
        combined.append(A)

    l = np.array([1.0], dtype=complex)
    r = np.concatenate([r1, r2]).astype(complex)
    return combined, l, r
