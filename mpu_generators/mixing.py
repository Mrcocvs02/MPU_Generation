# mpu_generators/mixing.py
r"""Combine independently certified MPU chains into a new certified chain.

All functions take and return ``(tensors, l_vec, r_vec)`` triples in the
package convention (``tensors`` is a list; use ``[A] * N`` for a single
translation-invariant tensor).

* :func:`stack` -- operator product :math:`U^{(1)}_N U^{(2)}_N`
  (tensor-first sampling paper, composition remark);
* :func:`capped_sum` -- one extra site carrying an (RR) pair,
  :math:`\Pi_1\otimes U^{(1)}_N+\Pi_2\otimes U^{(2)}_N` (thesis
  ``cor:mode1-readout`` applied to two chains);
* :func:`nest` -- alias of
  :meth:`~mpu_generators.uniform.UniformMPUGenerator.invariant_subspace`.

The naive sum :math:`\ell=\ell_1\oplus\ell_2` without the extra site is
not unitary in general (the cross terms :math:`U^{(1)\dagger}U^{(2)}` do not
cancel); the test suite contains this negative control.
"""

from __future__ import annotations

import numpy as np

from .base import _row_resolution_blocks


def stack(mpu1, mpu2, seed: int | None = None):
    r"""Operator product of two chains, :math:`U_N = U_N^{(1)}\,U_N^{(2)}`.

    Site tensors compose as operators (``mpu2`` acts first) and bond
    dimensions multiply, :math:`\chi=\chi_1\chi_2`; boundaries are Kronecker
    products. The result is unitary because both factors are.

    Parameters
    ----------
    mpu1, mpu2 : tuple
        ``(tensors, l_vec, r_vec)`` with equal length and physical dimension.
    seed : int, optional
        Unused (deterministic); kept for a uniform interface.

    Returns
    -------
    tensors, l_vec, r_vec
    """
    tensors1, l1, r1 = mpu1
    tensors2, l2, r2 = mpu2
    if len(tensors1) != len(tensors2):
        raise ValueError("stack requires equal-length chains.")
    d = tensors1[0].shape[0]
    if tensors2[0].shape[0] != d:
        raise ValueError(
            f"stack requires equal physical dimension d (got {d} and {tensors2[0].shape[0]})."
        )

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
    r"""Alias of :meth:`~mpu_generators.uniform.UniformMPUGenerator.invariant_subspace`.

    Parameters
    ----------
    sub_A, sub_l, sub_r : ndarray
        Translation-invariant MPU to embed.
    D_extra : int
        Additional bond dimension.
    seed : int, optional
        Seed for the embedding.
    generator : UniformMPUGenerator, optional
        Existing generator to use (a new one is created otherwise).

    Returns
    -------
    A, l_vec, r_vec : ndarray
    """
    from .uniform import UniformMPUGenerator

    gen = generator if generator is not None else UniformMPUGenerator(seed=seed or 0)
    return gen.invariant_subspace(sub_A, sub_l, sub_r, D_extra=D_extra, seed=seed)


def capped_sum(mpu1, mpu2, seed: int | None = None):
    r"""Capped sum: :math:`U_{N+1}=\Pi_1\otimes U_N^{(1)}+\Pi_2\otimes U_N^{(2)}`.

    A readout site is prepended whose row is an (RR) pair
    (:func:`~mpu_generators.base._row_resolution_blocks` with two targets),
    followed by the block-diagonal direct sum of the two chains. The
    identities :math:`\Pi_1^\dagger\Pi_1+\Pi_2^\dagger\Pi_2=1`,
    :math:`\Pi_1^\dagger\Pi_2=0` cancel the cross terms, so the result is
    unitary for any two unitary inputs.

    Parameters
    ----------
    mpu1, mpu2 : tuple
        ``(tensors, l_vec, r_vec)`` with equal length and physical dimension
        (bond dimensions may differ).
    seed : int, optional
        Seed for the Haar draws of the readout row (default 0).

    Returns
    -------
    tensors, l_vec, r_vec
        ``N + 1`` tensors, ``l_vec = (1)``.

    Notes
    -----
    The readout blocks are stored without the transpose used elsewhere, so the
    site applies :math:`\Pi_i^\top`; the pair :math:`\{\Pi_i^\top\}` satisfies
    (RR) as well (by the mirror identities), so unitarity is unaffected.
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
