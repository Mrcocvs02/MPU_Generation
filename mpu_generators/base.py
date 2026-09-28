# mpu_generators/base.py
r"""Certified random generators of open-boundary Matrix Product Unitaries.

Every generator returns site tensors ``A[p_in, p_out, bond_L, bond_R]`` and
boundary vectors ``(l, r)`` such that the contraction

.. math::
    U_N = \sum_{v_0,\dots,v_N} \ell_{v_0} r_{v_N}\,
          A[v_0,v_1]\otimes\cdots\otimes A[v_{N-1},v_N]

is exactly unitary for every :math:`N`, certified by an algebraic identity of
the local data (thesis, Ch. "Generating MPUs"; M. Vassallo, "Tensor-first
sampling of exact matrix-product unitaries"):

* **Mode I (row resolution)** -- :meth:`MPUGenerator.branching`;
* **Mode II (fusion closure)** -- :meth:`MPUGenerator.fusion` with the
  cyclic, nested-flag, matrix-unit and group-convolution skeletons;
* **Mode II-B (bond-resolved fusion)** -- :meth:`MPUGenerator.fusion_bond_resolved`;
* **grafted composition** -- :meth:`MPUGenerator.grafted`.

Conventions
-----------
Index order ``A[p_in, p_out, bond_L, bond_R]`` (as in :mod:`TreeMPU`).
The physical block stored at ``A[:, :, v, u]`` is the *transpose* of the
operator :math:`A[v,u]` (rows = output) used in the thesis. Boundary vectors
are contracted without complex conjugation. The package depends on
:mod:`TreeMPU` only for :func:`~TreeMPU.utils.block_mpu_tensors`,
:func:`~TreeMPU.utils.compute_chain_boundary_operators` and the dense
reference in :mod:`TreeMPU.checks`.
"""

from __future__ import annotations

import math
from itertools import permutations, product

import numpy as np
from TreeMPU.utils import (
    block_mpu_tensors,
    compute_chain_boundary_operators,
)
from TreeMPU.utils import satisfies_assumption_1 as _satisfies_assumption_1

#: Largest blocked tensor (d**L, d**L, D, D) complex128 that blocking may build.
#: Every attempt costs memory exponential in L (d = 3, L = 8 is ~6 GB), so the
#: size is checked before allocation, for explicit and automatic blocking.
_AUTO_BLOCKING_MAX_BYTES = 512 * 1024 * 1024


# =====================================================================
# Haar sampling
# =====================================================================


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a Haar-distributed unitary of size ``dim``.

    QR decomposition of a complex Ginibre matrix :math:`Z=QR`, with the phases
    of :math:`\mathrm{diag}(R)` absorbed into :math:`Q` so that the result is
    Haar distributed (Mezzadri, Notices AMS 54, 592 (2007)). Consumes exactly
    two ``standard_normal((dim, dim))`` draws from ``rng``.
    """
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R)
    ph = ph / np.maximum(np.abs(ph), 1e-300)
    return (Q * ph[None, :]).astype(complex)


# =====================================================================
# Mode I: branching (row resolution)
# =====================================================================


def _row_resolution_blocks(
    d: int, targets: list[int], rng: np.random.Generator
) -> dict[int, np.ndarray]:
    r"""Sample one row of a Mode I tensor obeying row resolution (RR).

    .. math::
        \sum_u A[v,u]^\dagger A[v,u] = 1_d, \qquad
        A[v,u]^\dagger A[v,u'] = 0 \quad (u \neq u'),              \tag{RR}

    together with the mirror identities for :math:`A[v,u]A[v,u']^\dagger`
    (thesis ``thm:mode1-unitarity``; paper, "row-resolution identity" in
    the section on exact certification mechanisms).

    Recipe: draw :math:`V, W` Haar in :math:`U(d)`, let :math:`P_i` be the
    orthogonal projector onto the :math:`i`-th consecutive block of columns
    of :math:`W` (block sizes as equal as possible, so the rank partition is
    balanced), and set :math:`A[v,u_i] = V P_i W^\dagger`. Both halves of (RR)
    hold for every draw; no rejection is needed.

    Parameters
    ----------
    d : int
        Physical dimension (or the enlarged dimension of a readout site).
    targets : list of int
        Target vertices :math:`u_i` of the outgoing edges of this row.
    rng : numpy.random.Generator
        Consumes two Haar draws (:math:`V` then :math:`W`).

    Returns
    -------
    dict[int, ndarray]
        ``{u_i: A[v, u_i]}`` as :math:`d\times d` matrices acting
        output <- input.

    Notes
    -----
    The resolved subspaces are column blocks of :math:`W` and :math:`A` is
    :math:`(VW)E_i(W^2)^\dagger` with coordinate projectors :math:`E_i`.
    The joint law of the blocks is therefore not the unitarily invariant one
    obtained from independent :math:`W_1 P_i W_2^\dagger`; both satisfy (RR),
    and all numerical results of the thesis/paper refer to this sampler.
    """
    m = len(targets)
    if not 1 <= m <= d:
        raise ValueError(f"A row needs 1 <= out-degree <= d outgoing edges; got {m} with d={d}.")
    base, rem = divmod(d, m)
    ranks = [base + (1 if i < rem else 0) for i in range(m)]

    V = _haar_unitary(d, rng)
    W = _haar_unitary(d, rng)

    blocks: dict[int, np.ndarray] = {}
    offset = 0
    for i, u in enumerate(targets):
        cols = W[:, offset : offset + ranks[i]]
        blocks[u] = V @ (cols @ cols.conj().T) @ W.conj().T
        offset += ranks[i]
    return blocks


def _check_row_resolution(A: np.ndarray, atol: float = 1e-9) -> None:
    r"""Check (RR) on every populated row of ``A[p_in, p_out, bond_L, bond_R]``.

    The stored block is the transpose of the operator,
    ``A[:, :, v, u] = A_op[v, u].T``, so
    :math:`\sum_u A_{\rm op}^\dagger A_{\rm op} = (\sum_u B B^\dagger)^\top`
    with ``B = A[:, :, v, u]``; the residuals computed below are therefore
    identical to those of (RR) itself.

    Raises
    ------
    RuntimeError
        If completeness or pairwise orthogonality fails by more than ``atol``
        (Frobenius norm) at some vertex.
    """
    d, _, D, _ = A.shape
    eye = np.eye(d, dtype=complex)
    for v in range(D):
        used = [u for u in range(D) if np.any(A[:, :, v, u])]
        if not used:
            continue
        acc = sum(A[:, :, v, u] @ A[:, :, v, u].conj().T for u in used)
        err = float(np.linalg.norm(acc - eye))
        if err > atol:
            raise RuntimeError(
                f"(RR) completeness failed at vertex {v}: ||sum_u A A^dag - 1|| = {err:.3e}"
            )
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                cross = float(np.linalg.norm(A[:, :, v, used[i]] @ A[:, :, v, used[j]].conj().T))
                if cross > atol:
                    raise RuntimeError(
                        f"(RR) orthogonality failed at vertex {v}: "
                        f"||A[{used[i]}] A[{used[j]}]^dag|| = {cross:.3e}"
                    )


def _mode1_site_tensor(
    D: int,
    edges: dict[int, list[int]],
    d: int,
    rng: np.random.Generator,
    identity_rows: tuple[int, ...] = (),
) -> np.ndarray:
    r"""Build one Mode I site tensor ``A[p_in, p_out, bond_L, bond_R]``.

    Parameters
    ----------
    D : int
        Bond dimension (number of vertices of the virtual digraph).
    edges : dict[int, list[int]]
        Out-neighbours of each vertex; row :math:`v` gets one (RR) row with
        these targets (see :func:`_row_resolution_blocks`).
    d : int
        Physical dimension.
    rng : numpy.random.Generator
        Consumed row by row, in the iteration order of ``edges``.
    identity_rows : tuple of int
        Vertices whose row is a single identity self-loop (no randomness).

    Returns
    -------
    ndarray
        Shape ``(d, d, D, D)``; entry ``[:, :, v, u]`` stores the transposed
        operator :math:`A[v,u]^\top` (index order ``p_in, p_out``).
    """
    A = np.zeros((d, d, D, D), dtype=complex)
    for v, targets in edges.items():
        if not targets:
            continue
        if v in identity_rows:
            if targets != [v]:
                raise ValueError(f"identity_rows entry {v} must carry exactly one self-loop.")
            A[:, :, v, v] = np.eye(d, dtype=complex)
            continue
        for u, block in _row_resolution_blocks(d, targets, rng).items():
            A[:, :, v, u] = block.T
    return A


def _random_out_regular_digraph(
    D: int, d: int, rng: np.random.Generator, out_degree: int | None = None
) -> dict[int, list[int]]:
    """Random out-regular digraph on ``D`` vertices.

    Every vertex gets ``m = min(out_degree or d, d, D)`` distinct targets,
    chosen uniformly without replacement (self-loops allowed). The cap at
    ``d`` is forced by (RR): a row resolves at most ``d`` mutually orthogonal
    branches. Cycles are allowed; the Mode I certificate needs no acyclicity.
    """
    m = min(d, D) if out_degree is None else min(out_degree, d, D)
    return {v: sorted(int(t) for t in rng.choice(D, size=m, replace=False)) for v in range(D)}


def _even_split(items: list, k: int) -> list[list]:
    """Partition ``items`` into ``k`` contiguous, size-balanced groups (order preserved)."""
    n = len(items)
    base, rem = divmod(n, k)
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    out = []
    offset = 0
    for size in sizes:
        out.append(items[offset : offset + size])
        offset += size
    return out


def _combined_readout_levels(readout: list[int], d: int) -> list[list[list[int]]]:
    r"""
    Group structure for combined readout over ``D_aux = len(readout)``
    vertices at physical dimension ``d`` (thesis ``rem:cascaded-readout``,
    ``D_aux > d`` case): a $d$-ary tree with ``D_aux`` leaves, built by repeatedly
    splitting every group of size > 1 into up to ``d`` roughly-even
    sub-groups; a group already at size 1 passes through unchanged (its
    own trivial, still (RR)-satisfying, out-degree-1 row) rather than
    being left to grow stale. Splitting always preserves the relative
    order of elements across groups, so the final level's groups are
    exactly ``[[readout[0]], [readout[1]], ..., [readout[-1]]]`` in the
    original order -- no leaf is ever left unassigned or relabeled, so
    unlike a naive fixed-depth ``d**m``-leaf tree, no padding-by-reusing-
    a-label step is needed at all (see :func:`_combined_readout_sites`).

    Returns ``levels``, ``levels[0] == [list(readout)]`` (all of it,
    unresolved) and ``levels[-1]`` all singletons; ``len(levels) - 1`` is
    the number of extra physical sites the cascade needs
    (``m = ceil(log_d(D_aux))``, forced to >= 1 even when ``D_aux == 1``
    so the cascade still appends exactly one site in that case, matching
    the ``D_aux <= d`` construction below at ``m = 1``).
    """
    levels = [[list(readout)]]
    while len(levels) == 1 or any(len(g) > 1 for g in levels[-1]):
        prev = levels[-1]
        new_level = []
        for g in prev:
            if len(g) == 1:
                new_level.append(g)
            else:
                k = min(d, len(g))
                new_level.extend(_even_split(g, k))
        levels.append(new_level)
    return levels


def _combined_readout_sites(
    readout: list[int], d: int, bond_dim: int, rng: np.random.Generator
) -> list[np.ndarray]:
    r"""Build the extra sites of a combined readout over ``readout`` (any ``D_aux``).

    Each level ``k -> k+1`` of :func:`_combined_readout_levels` becomes one
    site: bond ``L`` = number of groups at level ``k``, bond ``R`` = number
    of groups at level ``k + 1`` (or ``bond_dim`` at the last level, since
    that is what must plug into the rest of the digraph chain -- the
    populated columns are exactly ``readout``, everything else zero). Every row is an independent
    (RR) row from :func:`_row_resolution_blocks`, so the cascade is
    certified by the same Mode I theorem as the bulk chain.

    Returns
    -------
    list of ndarray
        Sites in level order; ``sites[0]`` has ``bond_L = 1`` (pairs with
        :math:`\ell=(1)`). Callers append ``list(reversed(sites))`` because
        :func:`_finalize_chain` reverses the whole list.
    """
    levels = _combined_readout_levels(readout, d)
    m = len(levels) - 1
    sites = []
    for k in range(m):
        cur, nxt = levels[k], levels[k + 1]
        offset = 0
        child_ranges = []
        for g in cur:
            if len(g) == 1:
                child_ranges.append([offset])
                offset += 1
            else:
                kk = min(d, len(g))
                child_ranges.append(list(range(offset, offset + kk)))
                offset += kk
        bond_L = len(cur)
        bond_R = bond_dim if k == m - 1 else len(nxt)
        A = np.zeros((d, d, bond_L, bond_R), dtype=complex)
        for p, children in enumerate(child_ranges):
            blocks = _row_resolution_blocks(d, children, rng)
            for c, block in blocks.items():
                A[:, :, p, nxt[c][0] if k == m - 1 else c] = block.T
        sites.append(A)
    return sites


def _check_combined_readout_cascade(sites: list[np.ndarray], atol: float = 1e-9) -> None:
    """
    Verify (RR) at every row of every combined-readout cascade site
    (completeness + orthogonality), same row-isometry contraction pattern
    as :func:`_check_row_resolution` for the ``p_in``-leading convention.
    """
    for A in sites:
        d, _, bond_L, bond_R = A.shape
        for p in range(bond_L):
            used = [c for c in range(bond_R) if np.any(A[:, :, p, c])]
            if not used:
                continue
            acc = sum(A[:, :, p, c] @ A[:, :, p, c].conj().T for c in used)
            err = float(np.linalg.norm(acc - np.eye(d)))
            if err > atol:
                raise RuntimeError(
                    f"(RR) completeness failed in combined-readout cascade at row {p}: {err:.3e}"
                )
            for i in range(len(used)):
                for j in range(i + 1, len(used)):
                    cross = float(
                        np.linalg.norm(A[:, :, p, used[i]] @ A[:, :, p, used[j]].conj().T)
                    )
                    if cross > atol:
                        raise RuntimeError(
                            f"(RR) orthogonality failed in combined-readout "
                            f"cascade at row {p}: {cross:.3e}"
                        )


# =====================================================================
# Mode II: fusion closure -- three named skeletons
# =====================================================================


def _cyclic_character_channels(d: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""Cyclic skeleton: :math:`y_g = W\,\mathrm{diag}(\omega^{0\cdot g},\dots,\omega^{(d-1)g})\,W^\dagger`.

    Here :math:`\omega=e^{2\pi i/M}`, :math:`M` = ``n_channels``,
    :math:`g=0,\dots,M-1`, and :math:`W` is one Haar unitary. The channels
    form a unitary representation of :math:`\mathbb Z_M` and close
    monomially, :math:`y_u^\dagger y_v = y_{(v-u)\bmod M}`.

    Returns
    -------
    ndarray
        Shape ``(n_channels, d, d)``.

    Notes
    -----
    This is a *group-convolution* skeleton in the sense of the thesis and
    of the tensor-first sampling paper (regular representation, channels
    Hilbert-Schmidt orthogonal) only when :math:`M` divides :math:`d`: then
    every character appears :math:`d/M` times and
    :math:`\operatorname{Tr} y_g = 0` for :math:`g\neq 0`. For
    :math:`M\nmid d` the family is still exactly fusion-closed and unitary,
    but :math:`\operatorname{Tr} y_g\neq 0` and the group-convolution
    causal results (e.g. :math:`S(\ell)=d^2(1-\sum_g|\ell_g|^4)`) do not
    apply. For :math:`d=M` the ``"cyclic"`` and ``"group"`` (``Zn``)
    skeletons have the same distribution.
    """
    omega = np.exp(2j * np.pi / n_channels)
    ks = np.arange(d)
    W = _haar_unitary(d, rng)
    return np.stack([W @ np.diag(omega ** (ks * g)) @ W.conj().T for g in range(n_channels)])


#: Backward-compatible alias (the function builds the cyclic skeleton).
_group_convolution_channels = _cyclic_character_channels


def _nested_flag_channels(d: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""Nested-flag skeleton: :math:`y_a = P_a`, the projector onto the last :math:`d-a` columns of a Haar :math:`W`.

    :math:`a = 0,\dots,M-1` with :math:`M` = ``n_channels``, so :math:`y_0 = 1_d`
    and :math:`\operatorname{rank} y_a = d-a`. The flag is nested
    (:math:`P_{a}\ge P_{a+1}`) and closes monomially,
    :math:`y_u^\dagger y_v = y_{\max(u,v)}`. For :math:`d=2` with a fixed
    basis this is the MCZ / diagonal-phase mechanism (thesis
    ``sec:beyond-qca``).

    Raises
    ------
    ValueError
        If ``n_channels > d``.
    """
    if n_channels > d:
        raise ValueError(
            f"nested-flag skeleton needs n_channels <= d (got n_channels={n_channels}, d={d})."
        )
    W = _haar_unitary(d, rng)
    return np.stack([W[:, a:] @ W[:, a:].conj().T for a in range(n_channels)])


def _matrix_unit_labels(m: int) -> list[tuple[int, int] | None]:
    r"""Channel labels of the matrix-units skeleton.

    Index 0 is a separate identity channel (label ``None``), followed by the
    :math:`m^2` matrix units :math:`(i,j)`, :math:`i,j=0,\dots,m-1`. The
    separate identity channel is essential: with the identity written as
    :math:`\sum_i y_{(i,i)}` instead, :math:`\sum_i y_{(i,i)}^{\otimes 2}\neq 1_d^{\otimes 2}`
    and no boundary gives a unitary for :math:`N\ge 2`.
    """
    return [None] + [(i, j) for i in range(m) for j in range(m)]


def _matrix_unit_channels(
    d: int, m: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int] | None]]:
    r"""Matrix-units channels: :math:`y_0=1_d` and :math:`y_{(i,j)} = w_i w_j^\dagger`.

    :math:`w_i` are the first :math:`m` columns of one Haar :math:`W`. The
    channels satisfy (FC), checked by :func:`_check_matrix_unit_closure`:

    .. math::
        y_0^\dagger y_v = y_v, \quad y_u^\dagger y_0 = y_{(j,i)}
        \ (u=(i,j)), \quad y_0^\dagger y_0 = y_0, \quad
        y_{(i,j)}^\dagger y_{(k,l)} = \delta_{ik}\, y_{(j,l)}.

    Returns
    -------
    y : ndarray
        Shape ``(1 + m**2, d, d)``, ``y[0] = 1_d``.
    labels : list
        From :func:`_matrix_unit_labels`.

    Raises
    ------
    ValueError
        If ``m > d``.
    """
    if m > d:
        raise ValueError(f"matrix-units skeleton needs m <= d (got m={m}, d={d}).")
    W = _haar_unitary(d, rng)
    w = [W[:, [i]] for i in range(m)]
    labels = _matrix_unit_labels(m)
    y = np.stack([np.eye(d, dtype=complex)] + [w[i] @ w[j].conj().T for (i, j) in labels[1:]])
    return y, labels


def _matrix_unit_boundary(m: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary on the matrix-units admissible manifold.

    With target :math:`e_0=1`, :math:`e_{(p,q)}=0` and
    :math:`\Lambda_{ij}:=\ell_{(i,j)}`, the boundary condition (B) reduces to
    :math:`\Lambda+\Lambda^\dagger+\Lambda^\dagger\Lambda=0`, i.e.
    :math:`V:=\Lambda+1_m` is unitary. The manifold is therefore
    :math:`\ell_0=1`, :math:`\ell_{(i,j)}=(V-1_m)_{ij}`, :math:`V\in U(m)`, and
    :math:`V` is drawn from the Haar measure.
    """
    labels = _matrix_unit_labels(m)
    V = _haar_unitary(m, rng)
    l_vec = np.zeros(len(labels), dtype=complex)
    l_vec[0] = 1.0
    for idx, lab in enumerate(labels):
        if lab is not None:
            i, j = lab
            l_vec[idx] = V[i, j] - (1.0 if i == j else 0.0)
    return l_vec


def _check_fusion_closure_general(y: np.ndarray, gamma: np.ndarray, atol: float = 1e-9) -> None:
    r"""
    Verify the general fusion closure condition (FC),
    ``y[u]^dagger y[v] == sum_w gamma[w,u,v] y[w]``, for an arbitrary
    (not necessarily cyclic) skeleton with structure constants ``gamma``
    of shape ``(n_channels, n_channels, n_channels)`` (indexed ``[w, u, v]``).
    """
    n = y.shape[0]
    for u in range(n):
        for v in range(n):
            lhs = y[u].conj().T @ y[v]
            rhs = np.einsum("w,wij->ij", gamma[:, u, v], y)
            err = float(np.linalg.norm(lhs - rhs))
            if err > atol:
                raise RuntimeError(
                    f"(FC) failed at (u={u}, v={v}): ||y[u]^dag y[v] - "
                    f"sum_w gamma[w,u,v] y[w]|| = {err:.3e}"
                )


def _check_fusion_closure(y: np.ndarray, atol: float = 1e-9) -> None:
    """Verify (FC) for the monomial cyclic skeleton: ``y[u]^dag y[v] == y[(v-u) mod M]``."""
    M = y.shape[0]
    for u in range(M):
        for v in range(M):
            err = float(np.linalg.norm(y[u].conj().T @ y[v] - y[(v - u) % M]))
            if err > atol:
                raise RuntimeError(
                    f"(FC) failed at (u={u}, v={v}): ||y[u]^dag y[v] - y[(v-u) mod M]|| = {err:.3e}"
                )


def _check_nested_flag_closure(y: np.ndarray, atol: float = 1e-9) -> None:
    """Verify (FC) for the nested-flag skeleton: ``y[u]^dag y[v] == y[max(u,v)]``."""
    n = y.shape[0]
    for u in range(n):
        for v in range(n):
            err = float(np.linalg.norm(y[u].conj().T @ y[v] - y[max(u, v)]))
            if err > atol:
                raise RuntimeError(
                    f"(FC) failed [nested-flag] at (u={u}, v={v}): "
                    f"||y[u]^dag y[v] - y[max(u,v)]|| = {err:.3e}"
                )


def _check_matrix_unit_closure(
    y: np.ndarray, labels: list[tuple[int, int] | None], atol: float = 1e-9
) -> None:
    r"""
    Verify (FC) for matrix units WITH the separate identity channel
    (label ``None`` at index 0): ``y_0^dag y_v = y_v``,
    ``y_u^dag y_0 = y_{(j,i)}`` (``u=(i,j)``), ``y_0^dag y_0 = y_0``, and
    ``y_{(i,j)}^dag y_{(k,l)} = delta_ik y_{(j,l)}`` for the rest.
    """
    index = {lab: idx for idx, lab in enumerate(labels)}

    def expected(a_lab, b_lab):
        if a_lab is None and b_lab is None:
            return index[None]
        if a_lab is None:
            return index[b_lab]
        if b_lab is None:
            i, j = a_lab
            return index[(j, i)]
        i, j = a_lab
        k, l = b_lab
        return index[(j, l)] if i == k else None

    for a, a_lab in enumerate(labels):
        for b, b_lab in enumerate(labels):
            lhs = y[a].conj().T @ y[b]
            idx = expected(a_lab, b_lab)
            rhs = y[idx] if idx is not None else np.zeros_like(lhs)
            err = float(np.linalg.norm(lhs - rhs))
            if err > atol:
                raise RuntimeError(
                    f"(FC) failed [matrix-units] at ({a_lab},{b_lab}): ||...|| = {err:.3e}"
                )


def _check_channel_span_full_rank(y: np.ndarray, n_channels: int, tol: float = 1e-9) -> None:
    """
    Verify the ``n_channels`` sampled channels are linearly independent (as
    vectors in ``M_d(C)``), i.e. that the caller-facing ``bond_dim`` will not
    be silently reduced by MPU canonical form.
    """
    flat = y.reshape(n_channels, -1)
    rank = int(np.linalg.matrix_rank(flat, tol=tol))
    if rank < n_channels:
        raise RuntimeError(
            f"Sampled channels span only rank {rank} < n_channels="
            f"{n_channels}; the resulting chain's bond dimension would be "
            f"reduced by canonical form."
        )


def _flat_spectrum_boundary(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary for the cyclic skeleton: flat Fourier spectrum.

    With :math:`\gamma^w_{uv}=\delta_{w,(v-u)\bmod M}` the boundary condition
    (B) is :math:`|\mathrm{DFT}(\ell)_k|=1` for all :math:`k`. Uniform random
    phases are drawn and inverse-transformed (uniform measure on the torus
    :math:`T^M`).
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
    return np.fft.ifft(np.exp(1j * theta)).astype(complex)


def _check_flat_spectrum(l_vec: np.ndarray, atol: float = 1e-9) -> None:
    """Verify (B) for the cyclic skeleton: ``|DFT(l)[k]| = 1`` for every k."""
    err = float(np.max(np.abs(np.abs(np.fft.fft(l_vec)) - 1.0)))
    if err > atol:
        raise RuntimeError(f"(B) failed: max_k | |DFT(l)[k]| - 1 | = {err:.3e}")


def _nested_flag_boundary(bond_dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary on the nested-flag admissible manifold.

    For the closure :math:`\gamma^{\max(u,v)}_{uv}=1` with target
    :math:`e_w=\delta_{w,0}`, the boundary condition (B) is triangular.
    Level :math:`w=0` gives :math:`|\ell_0|^2=1`; level :math:`w>0` gives

    .. math::
        |\ell_w|^2 + 2\,\mathrm{Re}(\bar\ell_w S_w) = 0
        \iff |\ell_w + S_w| = |S_w|, \qquad S_w := \sum_{u<w}\ell_u .

    Choosing :math:`\ell_0=1` gives :math:`S_1=1`, and every level preserves
    :math:`|S_{w+1}|=|\ell_w+S_w|=|S_w|=1`. The manifold (with
    :math:`\ell_0=1` fixed) is thus the flat torus
    :math:`\ell_w=-S_w+e^{i\theta_w}`, :math:`\theta\in[0,2\pi)^{\chi-1}`;
    the angles are drawn uniformly. :math:`\theta_w=\arg S_w` gives the
    degenerate point :math:`\ell_w=0`; for :math:`\chi=2`, :math:`\theta=\pi`
    gives the MCZ boundary :math:`\ell=(1,-2)`.
    """
    l_vec = np.zeros(bond_dim, dtype=complex)
    l_vec[0] = 1.0 + 0.0j
    for w in range(1, bond_dim):
        S_w = np.sum(l_vec[:w])  # |S_w| = 1 exactly (see docstring)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        l_vec[w] = -S_w + abs(S_w) * np.exp(1j * theta)
    return l_vec


def _finite_group_table(group: str, n: int | list[int] = 0):
    r"""Multiplication table and irreducible representations of a small finite group.

    ``group``:
      - ``"Zn"``: cyclic group of order ``n`` (abelian). Irreps: the ``n``
        one-dimensional characters $\rho_k(g) = \omega^{kg}$,
        $\omega = e^{2\pi i/n}$ -- standard, works for any ``n``.
      - ``"Zn_product"``: direct product $Z_{n_1} \times \cdots \times
        Z_{n_k}$ (``n`` is a list of factor orders here, not a single
        int) -- *any* finite abelian group, by the fundamental theorem of
        finite abelian groups (every finite abelian group is a product of
        cyclic factors). Irreps are the products of per-factor characters,
        $\rho_{\vec k}(\vec g) = \prod_i \omega_i^{k_i g_i}$.
      - ``"S3"``: the symmetric group on 3 letters (dihedral of order 6,
        the smallest non-abelian group). ``n`` is ignored. Irreps: trivial
        (1-dim), sign (1-dim, +1 on the identity/two 3-cycles, -1 on the
        three transpositions), and the 2-dim "standard" representation
        (S3 as the symmetries of a triangle: 3-cycles are rotations by
        120 degrees, transpositions are reflections); the homomorphism
        property is re-checked by :func:`_check_group_representation`.

    Returns
    -------
    tuple
        ``(elements, mult, inv, identity, irreps)`` with ``irreps`` a list
        of ``(dim, [rho(g) for g in elements])`` pairs.

    Notes
    -----
    The tensor-first sampling paper states the group-convolution results for
    every finite group; this implementation provides the abelian groups and
    :math:`S_3` only (general groups need their irreducible representations
    as input).
    """
    if group == "Zn_product":
        factors = list(n)
        if len(factors) < 1 or any(f < 1 for f in factors):
            raise ValueError(
                f"group='Zn_product' needs a non-empty list of factors >= 1, got {factors}."
            )
        elements = list(product(*(range(f) for f in factors)))
        mult = lambda g, h: tuple((gi + hi) % f for gi, hi, f in zip(g, h, factors))
        inv = lambda g: tuple((-gi) % f for gi, f in zip(g, factors))
        identity = tuple(0 for _ in factors)
        omegas = [np.exp(2j * np.pi / f) for f in factors]
        irreps = []
        for k_vec in product(*(range(f) for f in factors)):

            def rho_g(g, k_vec=k_vec):
                phase = 1.0 + 0.0j
                for ki, gi, om in zip(k_vec, g, omegas):
                    phase *= om ** (ki * gi)
                return np.array([[phase]], dtype=complex)

            irreps.append((1, [rho_g(g) for g in elements]))
        return elements, mult, inv, identity, irreps

    if group == "Zn":
        if n < 1:
            raise ValueError(f"group='Zn' needs n >= 1, got n={n}.")
        elements = list(range(n))
        mult = lambda g, h: (g + h) % n
        inv = lambda g: (-g) % n
        identity = 0
        omega = np.exp(2j * np.pi / n)
        irreps = [
            (1, [np.array([[omega ** (k * g)]], dtype=complex) for g in elements]) for k in range(n)
        ]
        return elements, mult, inv, identity, irreps

    if group == "S3":
        elements = list(permutations(range(3)))
        mult = lambda g, h: tuple(g[h[i]] for i in range(3))
        inv = lambda g: tuple(sorted(range(3), key=lambda i: g[i]))
        identity = (0, 1, 2)

        def _parity(perm):
            seen = [False] * 3
            sign = 1
            for i in range(3):
                if not seen[i]:
                    j, clen = i, 0
                    while not seen[j]:
                        seen[j] = True
                        j = perm[j]
                        clen += 1
                    if clen % 2 == 0:
                        sign *= -1
            return sign

        trivial = [np.array([[1.0]], dtype=complex) for _ in elements]
        sign = [np.array([[float(_parity(g))]], dtype=complex) for g in elements]

        # Standard 2-dim irrep: realize S3 as the symmetries of a triangle.
        # r = 3-cycle "rotate by 120 deg", s = a transposition "reflect".
        # Every element is r^k or s.r^k for k=0,1,2 -- classify each
        # element by that (rotation count, is-reflection) coordinate, then
        # assign the corresponding rotation/reflection 2x2 matrix, M(g) =
        # R^k or S @ R^k so that M is a genuine homomorphism,
        # M(mult(g,h)) = M(g) @ M(h) -- mult(g,h) = g compose h here, so
        # the element "s compose r^k" must be walked as mult(cur, r), not
        # mult(r, cur) (composing r on the WRONG side silently relabels
        # two of the three reflections against each other, since s.r^k =
        # r^{-k}.s in a dihedral group -- caught by
        # :func:`_check_group_representation`, not assumed correct).
        r = (1, 2, 0)
        s = (1, 0, 2)
        coord = {}
        cur = identity
        for k in range(3):
            coord[cur] = (k, False)
            cur = mult(r, cur)
        cur = s
        for k in range(3):
            coord[cur] = (k, True)
            cur = mult(cur, r)

        theta = 2 * np.pi / 3
        c, sn = np.cos(theta), np.sin(theta)
        R = np.array([[c, -sn], [sn, c]])
        S = np.array([[1.0, 0.0], [0.0, -1.0]])
        standard = []
        for g in elements:
            k, is_reflection = coord[g]
            M = np.linalg.matrix_power(R, k)
            if is_reflection:
                M = S @ M
            standard.append(M.astype(complex))

        irreps = [(1, trivial), (1, sign), (2, standard)]
        return elements, mult, inv, identity, irreps

    raise ValueError(f"Unknown group {group!r}; expected 'Zn', 'Zn_product' or 'S3'.")


def _check_group_representation(elements, mult, irreps, atol: float = 1e-9) -> None:
    r"""Check that every irrep is unitary and a homomorphism, :math:`\rho(g)\rho(h)=\rho(gh)`."""
    idx = {g: i for i, g in enumerate(elements)}
    for dim, rho in irreps:
        for gi, g in enumerate(elements):
            M = rho[gi]
            err_u = float(np.linalg.norm(M.conj().T @ M - np.eye(dim)))
            if err_u > atol:
                raise RuntimeError(
                    f"Irrep (dim {dim}) is not unitary at g={g}: "
                    f"||rho(g)^dag rho(g) - I|| = {err_u:.3e}"
                )
            for hi, h in enumerate(elements):
                lhs = M @ rho[hi]
                rhs = rho[idx[mult(g, h)]]
                err_h = float(np.linalg.norm(lhs - rhs))
                if err_h > atol:
                    raise RuntimeError(
                        f"Irrep (dim {dim}) is not a homomorphism at "
                        f"(g={g}, h={h}): ||rho(g)rho(h) - rho(gh)|| = "
                        f"{err_h:.3e}"
                    )


def _group_regular_channels(elements, mult) -> np.ndarray:
    r"""Group-convolution channels: the left-regular representation of :math:`G`.

    Physical dimension :math:`d=|G|`, one channel per element,
    :math:`y_g|h\rangle=|gh\rangle`. Closure is monomial for every finite
    group, :math:`y_g^\dagger y_h = y_{g^{-1}h}` (structure constants from
    :func:`_group_fusion_gamma`).
    """
    D = len(elements)
    idx = {g: i for i, g in enumerate(elements)}
    y = np.zeros((D, D, D), dtype=complex)
    for gi, g in enumerate(elements):
        for hi, h in enumerate(elements):
            y[gi, idx[mult(g, h)], hi] = 1.0
    return y


def _group_fusion_gamma(elements, mult, inv) -> np.ndarray:
    """Structure constants for :func:`_group_regular_channels`: ``gamma^w_{uv} = delta_{w, g_u^{-1} g_v}``."""
    D = len(elements)
    idx = {g: i for i, g in enumerate(elements)}
    gamma = np.zeros((D, D, D))
    for u in range(D):
        for v in range(D):
            w = idx[mult(inv(elements[u]), elements[v])]
            gamma[w, u, v] = 1.0
    return gamma


def _group_fourier_boundary(elements, irreps, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary on the group-convolution (Peter-Weyl) admissible manifold.

    For a finite group, every function $f:G\to\mathbb C$ has a Fourier
    transform at each irrep $\rho$, $\hat f(\rho) = \sum_g f(g)\rho(g)$,
    invertible as $f(g) = \frac1{|G|}\sum_\rho d_\rho
    \operatorname{Tr}(\rho(g)^\dagger \hat f(\rho))$ (Peter-Weyl). Writing
    out (B), $\sum_g \overline{l(g)}\,l(gw) = \delta_{w,e}$, as a Fourier
    transform in $w$ shows the left side's transform at $\rho$ equals
    $\hat l(\rho)^\dagger \hat l(\rho)$, while $\delta_e$'s transform at
    $\rho$ is $\rho(e) = 1_{d_\rho}$ -- so (B) holds *iff* $\hat l(\rho)$
    is unitary at *every* irrep (exactly "flat spectrum", generalized from
    1x1 characters to $d_\rho\times d_\rho$ blocks). The manifold is
    $\prod_\rho U(d_\rho)$ (paper, Peter-Weyl ensemble); one Haar unitary
    $U_\rho$ is drawn per irrep, in the order of ``irreps``, and transformed
    back. The result is re-checked by the direct double sum in
    :func:`_check_group_boundary`.
    """
    G = len(elements)
    Us = [_haar_unitary(dim, rng) for dim, _ in irreps]
    l_vec = np.zeros(G, dtype=complex)
    for gi in range(G):
        acc = 0.0 + 0.0j
        for (dim, rho), U in zip(irreps, Us):
            acc += dim * np.trace(rho[gi].conj().T @ U)
        l_vec[gi] = acc / G
    return l_vec


def _check_group_boundary(
    l_vec: np.ndarray, elements, mult, idx, identity, atol: float = 1e-9
) -> None:
    """Brute-force (not Fourier-based) check of (B): ``sum_g conj(l(g)) l(gw) == delta_{w,identity}`` for every w."""
    for wi, w in enumerate(elements):
        target = 1.0 if w == identity else 0.0
        S = 0.0 + 0.0j
        for gi, g in enumerate(elements):
            S += np.conj(l_vec[gi]) * l_vec[idx[mult(g, w)]]
        err = abs(S - target)
        if err > atol:
            raise RuntimeError(f"(B) failed [group] at w={w}: |S(w) - target| = {err:.3e}")


# =====================================================================
# Common tail: site-order reversal, blocking, chain verification
# =====================================================================


def _finalize_chain(
    sites: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    blocking: int,
    verify: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    r"""Order, block and (optionally) sanity-check a freshly built chain.

    Every ``build_sites`` builder appends sites in construction order, which
    is the *reverse* of the physical left-to-right order of the chain
    (the thesis writes the product as :math:`\ell A_N \cdots A_1 r`); the
    list is reversed here, then grouped into blocks of ``blocking`` sites.
    Sites are already in the package convention
    ``A[p_in, p_out, bond_L, bond_R]``.

    Parameters
    ----------
    sites : list of ndarray
        Site tensors in construction order.
    l_vec, r_vec : ndarray
        Boundary vectors (copied, never modified in place).
    blocking : int
        Number of physical sites merged into one tensor.
    verify : bool
        Run the numerical checks described below.

    Notes
    -----
    Exact unitarity for every :math:`N` is guaranteed by the algebraic
    certificate each mode checks when its site tensor is built
    (row resolution, fusion closure, group-boundary condition). The checks
    run here are independent, *numerical* safeguards against implementation
    errors:

    * the normalization :math:`\operatorname{Tr}(U^\dagger U)/d^N = 1`
      (one left sweep, any :math:`N`; necessary but not sufficient for
      unitarity);
    * when :math:`d^N \le` ``_DENSE_VERIFY_MAX_DIM``, the exact condition
      :math:`\|U^\dagger U - I\|_F \le 10^{-8}\sqrt{d^N}` on the dense
      matrix.
    """
    tensors = block_mpu_tensors(list(reversed(sites)), blocking)
    l_vec = l_vec.copy()
    r_vec = r_vec.copy()
    if verify:
        _verify_chain(tensors, l_vec, r_vec)
    return tensors, l_vec, r_vec


#: Tolerance of the numerical safeguards in :func:`_finalize_chain`.
_VERIFY_TOL = 1e-8
#: Largest Hilbert-space dimension d**N for which ``verify=True`` also
#: builds the dense unitary (2**10 x 2**10 complex = 16 MiB).
_DENSE_VERIFY_MAX_DIM = 2**10


def _verify_chain(tensors: list[np.ndarray], l_vec: np.ndarray, r_vec: np.ndarray) -> None:
    """Numerical safeguards run by ``verify=True`` (see :func:`_finalize_chain`).

    Raises
    ------
    RuntimeError
        If the normalization differs from 1, or (for small ``d**N``) the
        dense matrix is not unitary, within ``_VERIFY_TOL``.
    """
    _, _, residual = compute_chain_boundary_operators(tensors, l_vec, r_vec)
    err = abs(1.0 - complex(residual))
    if err > _VERIFY_TOL:
        raise RuntimeError(
            f"Generated MPU is not unitary: normalization Tr(U^dag U)/d^N = {residual} "
            f"deviates from 1 by {err:.3e}."
        )
    # Dense check; the dimension is computed from the shapes before any contraction.
    dim = 1
    for A in tensors:
        dim *= A.shape[0]
        if dim > _DENSE_VERIFY_MAX_DIM:
            return
    from TreeMPU.checks import dense_unitary_from_tensors

    U = dense_unitary_from_tensors(tensors, l_vec, r_vec)
    defect = np.linalg.norm(U.conj().T @ U - np.eye(dim)) / np.sqrt(dim)
    if defect > _VERIFY_TOL:
        raise RuntimeError(
            f"Generated MPU is not unitary: ||U^dag U - I||_F / sqrt(d^N) = {defect:.3e}."
        )


def _run_blocking(one_blocking, d, bond_dim, n_sites, n_blocks, blocking, max_blocking):
    """Call ``one_blocking(n_sites_total, L)`` for a fixed or automatic block size ``L``.

    With ``blocking="auto"``, block sizes ``L = 1, ..., max_blocking`` are
    tried in order and the first chain satisfying Assumption 1 is returned.
    The size of the blocked tensor, ``(d**L)**2 * bond_dim**2`` complex
    numbers, is checked *before* it is built; the search stops at the first
    ``L`` above ``_AUTO_BLOCKING_MAX_BYTES``; an explicit ``blocking=L`` above
    the cap raises ``ValueError``.

    Raises
    ------
    ValueError
        If no admissible ``L`` gives a chain satisfying Assumption 1.
    """
    if blocking == "auto":
        attempts = []
        for L in range(1, max_blocking + 1):
            sites_L = n_blocks * L if n_blocks is not None else n_sites
            if n_blocks is None and (n_sites % L or n_sites // L < 2):
                continue
            projected_bytes = (d**L) ** 2 * bond_dim**2 * 16  # complex128
            if projected_bytes > _AUTO_BLOCKING_MAX_BYTES:
                attempts.append(
                    f"L={L}: skipped, projected tensor "
                    f"{projected_bytes / 1e6:.0f} MB exceeds safety cap "
                    f"{_AUTO_BLOCKING_MAX_BYTES / 1e6:.0f} MB (larger L "
                    "only gets bigger, so the search stops here)"
                )
                break
            try:
                cand = one_blocking(sites_L, L)
            except ValueError as exc:
                attempts.append(f"L={L}: {exc}")
                continue
            if _satisfies_assumption_1(*cand):
                return cand
            attempts.append(f"L={L}: rank-deficient boundary operators")
        raise ValueError(
            f"No blocking <= {max_blocking} satisfies Assumption 1. Tried: " + "; ".join(attempts)
        )
    L = int(blocking)
    projected_bytes = (d**L) ** 2 * bond_dim**2 * 16  # complex128, checked before allocation
    if projected_bytes > _AUTO_BLOCKING_MAX_BYTES:
        raise ValueError(
            f"blocking={L} would build tensors of {projected_bytes / 1e6:.0f} MB "
            f"(> {_AUTO_BLOCKING_MAX_BYTES / 1e6:.0f} MB safety cap); use a smaller block size."
        )
    sites_L = n_blocks * L if n_blocks is not None else n_sites
    return one_blocking(sites_L, L)


# =====================================================================
# MPUGenerator
# =====================================================================


class MPUGenerator:
    r"""Certified random generator of open-boundary MPUs (site-dependent).

    Every method returns ``(tensors, l_vec, r_vec)``: a list of site tensors
    ``A[p_in, p_out, bond_L, bond_R]`` and two boundary vectors, such that

    .. math::
        U_N = \sum_{v_0,\dots,v_N} \ell_{v_0} r_{v_N}\,
              A[v_0,v_1]\otimes\cdots\otimes A[v_{N-1},v_N]

    is exactly unitary for every :math:`N`. The certificate is an algebraic
    identity of the local data checked when each tensor is built (row
    resolution for Mode I, fusion closure plus an admissible boundary for
    Mode II), not a finite-:math:`N` test.

    In this class every site draws its own Haar dressing;
    :class:`~mpu_generators.uniform.UniformMPUGenerator` reuses one site
    tensor at every site (translation invariance) and is otherwise
    identical.

    Parameters
    ----------
    seed : int
        Seed of the ``numpy.random.default_rng`` created at every call.
        Calling the same method twice with the same arguments returns the
        same MPU, and for uniform generators the site tensor does not depend
        on ``n_sites`` (the ensembles of the tensor-first sampling paper rely
        on this).
    verify : bool
        Run the algebraic checks at construction time, plus the numerical
        safeguards of :func:`_finalize_chain` on the assembled chain.

    Notes
    -----
    Sites are built in construction order and reversed at the end
    (:func:`_finalize_chain`). With per-site resampling, physical site
    :math:`j` of an :math:`N`-site chain therefore carries the
    :math:`(N-1-j)`-th draw: the chains for increasing :math:`N` extend a
    common sequence at the *left* end. For Mode I with ``readout``, the
    readout sites are drawn after the bulk sites, so they depend on
    :math:`N` in the resampled case.
    """

    #: ``False`` in :class:`UniformMPUGenerator`: draw one site tensor and reuse
    #: it at every site (translation invariance).
    _resample: bool = True

    def __init__(self, seed: int = 0, verify: bool = True):
        self.seed = seed
        self.verify = verify

    def _sites(self, make_site, n_sites: int, rng: np.random.Generator) -> list[np.ndarray]:
        if self._resample:
            return [make_site(rng) for _ in range(n_sites)]
        site = make_site(rng)
        return [site] * n_sites

    # -----------------------------------------------------------------
    # Blocking / auto-blocking driver (mode-agnostic)
    # -----------------------------------------------------------------

    def _generate(
        self,
        d: int,
        bond_dim: int,
        n_sites: int | None,
        n_blocks: int | None,
        blocking: int | str,
        max_blocking: int,
        build_sites,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Validate sizes, build the chain and apply (auto-)blocking.

        ``build_sites(n_sites_total, rng)`` returns ``(sites, l_vec, r_vec)``
        in construction order. With ``blocking="auto"`` the smallest block
        size ``L <= max_blocking`` whose blocked chain satisfies Assumption 1
        (full-rank boundary operators at every cut, required by the TreeMPU
        decomposition) is returned. The projected size of each blocked tensor
        is checked *before* it is built (``_AUTO_BLOCKING_MAX_BYTES``).
        """
        if (n_sites is None) == (n_blocks is None):
            raise ValueError("Give exactly one of n_sites or n_blocks.")
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}.")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")
        if n_blocks is not None and n_blocks < 2:
            raise ValueError(f"n_blocks must be >= 2, got {n_blocks}.")

        def _one_blocking(sites_L: int, L: int):
            if sites_L < 2:
                raise ValueError(f"n_sites must be >= 2, got {sites_L}.")
            if sites_L % L:
                raise ValueError(f"n_sites={sites_L} is not divisible by blocking={L}.")
            if sites_L // L < 2:
                raise ValueError(
                    f"n_sites={sites_L} with blocking={L} leaves "
                    f"{sites_L // L} tensor(s); need at least 2."
                )
            rng = np.random.default_rng(self.seed)
            sites, l_vec, r_vec = build_sites(sites_L, rng)
            return _finalize_chain(sites, l_vec, r_vec, L, self.verify)

        return _run_blocking(_one_blocking, d, bond_dim, n_sites, n_blocks, blocking, max_blocking)

    # -----------------------------------------------------------------
    # Mode I: branching
    # -----------------------------------------------------------------

    def branching(
        self,
        d: int = 2,
        bond_dim: int = 2,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
        readout: list[int] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Mode I (row resolution): a random digraph with Haar-resolved rows.

        The virtual bond indexes the ``bond_dim`` vertices of a random
        out-regular digraph (out-degree ``min(d, bond_dim)``, balanced rank
        partition); every row of every site satisfies (RR)
        (:func:`_row_resolution_blocks`). With single-vertex readout
        :math:`\ell=e_0`, :math:`r=(1,\dots,1)` the chain is unitary for every
        :math:`N` (thesis ``thm:mode1-unitarity``; paper ``thm:main1``).

        Parameters
        ----------
        d : int
            Local physical dimension.
        bond_dim : int
            Bond dimension :math:`\chi` (number of digraph vertices).
        n_sites, n_blocks : int, optional
            Give exactly one: total physical sites, or number of tensors
            after blocking.
        blocking : int or "auto"
            Physical sites per returned tensor (see :meth:`_generate`).
        max_blocking : int
            Largest block size tried by ``blocking="auto"``.
        readout : list of int, optional
            Combined readout over the vertex set ``readout``
            (thesis ``cor:mode1-readout``, ``rem:cascaded-readout``): appends
            :math:`m=\max(1,\lceil\log_d D_{\rm aux}\rceil)` extra sites, a
            :math:`d`-ary tree of (RR) rows that resolves every listed vertex
            exactly (:func:`_combined_readout_sites`); then :math:`\ell=(1)`.

        Returns
        -------
        tensors : list of ndarray
            Site tensors ``(d_eff, d_eff, D_L, D_R)``.
        l_vec, r_vec : ndarray
            Boundary vectors.
        """

        def build_sites(n_sites_total: int, rng: np.random.Generator):
            edges = _random_out_regular_digraph(bond_dim, d, rng)

            def make_site(r: np.random.Generator) -> np.ndarray:
                site_edges = (
                    edges if not self._resample else _random_out_regular_digraph(bond_dim, d, r)
                )
                return _mode1_site_tensor(bond_dim, site_edges, d, r)

            sites = self._sites(make_site, n_sites_total, rng)

            if self.verify:
                for A in sites[: (1 if not self._resample else n_sites_total)]:
                    _check_row_resolution(A)

            if readout is None:
                l_vec = np.zeros(bond_dim, dtype=complex)
                l_vec[0] = 1.0
            else:
                cascade_sites = _combined_readout_sites(readout, d, bond_dim, rng)
                if self.verify:
                    _check_combined_readout_cascade(cascade_sites)
                # cascade_sites[0] has bond_L = 1 and pairs with l_vec = (1).
                # Appended in reverse so that _finalize_chain's reversal puts
                # the cascade at the left end in level order.
                sites = sites + list(reversed(cascade_sites))
                l_vec = np.array([1.0], dtype=complex)

            r_vec = np.ones(bond_dim, dtype=complex)
            return sites, l_vec, r_vec

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites)

    # -----------------------------------------------------------------
    # Mode II: fusion closure
    # -----------------------------------------------------------------

    def fusion(
        self,
        d: int = 2,
        bond_dim: int = 2,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
        skeleton: str = "cyclic",
        group: str = "Zn",
        group_n: int | list[int] = 3,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Mode II (fusion closure): bond-diagonal tensors on a closed channel set.

        :math:`A[a,b]=\delta_{ab}\,y_a` with channels closing monomially,
        :math:`y_u^\dagger y_v\in\{0\}\cup\{y_w\}` (FC), and a boundary
        :math:`\ell` on the skeleton's admissible manifold (B), :math:`r=(1,\dots,1)`.
        Then :math:`U_N=\sum_u \ell_u\,y_u^{\otimes N}` is unitary for every
        :math:`N` (thesis ``thm:mode2-unitarity``; paper ``eq:fusion-closure``).
        Each site conjugates all channels by one Haar unitary (dressing).

        Parameters
        ----------
        d, bond_dim, n_sites, n_blocks, blocking, max_blocking
            As in :meth:`branching`; ``bond_dim`` is the number of channels.
        skeleton : {"cyclic", "nested_flag", "matrix_units", "group"}
            Channel family (below).
        group : {"Zn", "Zn_product", "S3"}
            Group for ``skeleton="group"`` (see :func:`_finite_group_table`).
        group_n : int or list of int
            Order of ``Zn``, or the factor orders of ``Zn_product``.

        Returns
        -------
        tensors, l_vec, r_vec
            As in :meth:`branching`.

        Notes
        -----
        ``skeleton``:
        - ``"cyclic"`` (default): :math:`\mathbb Z_M` characters
          (:func:`_cyclic_character_channels`), flat-Fourier boundary
          (:func:`_flat_spectrum_boundary`). Needs ``bond_dim <= d``. A
          group-convolution skeleton only when ``bond_dim`` divides ``d``.
        - ``"nested_flag"``: nested projectors, closure :math:`\max(u,v)`
          (:func:`_nested_flag_channels`, :func:`_nested_flag_boundary`);
          contains MCZ / diagonal phase. Needs ``bond_dim <= d``.
        - ``"matrix_units"``: identity channel plus :math:`m^2` matrix units
          :math:`y_{(i,j)}=w_iw_j^\dagger` (:func:`_matrix_unit_channels`,
          :func:`_matrix_unit_boundary`); contains subspace rotations.
          ``bond_dim = 1 + m**2`` with ``m <= d`` (2, 5, 10, 17, ...).
        - ``"group"``: group convolution, the left-regular representation
          of a finite group (:func:`_group_regular_channels`), boundary on
          the Peter-Weyl manifold :math:`\prod_\rho U(d_\rho)`
          (:func:`_group_fourier_boundary`); ``d = bond_dim = |G|``.
          ``group="Zn_product"`` covers every finite abelian group,
          ``group="S3"`` the smallest non-abelian one.
        """
        if skeleton == "cyclic":
            if bond_dim > d:
                raise ValueError(
                    f"skeleton='cyclic' needs d >= bond_dim (got d={d}, bond_dim={bond_dim})."
                )
        elif skeleton == "nested_flag":
            if bond_dim > d:
                raise ValueError(
                    f"skeleton='nested_flag' needs d >= bond_dim (got d={d}, bond_dim={bond_dim})."
                )
        elif skeleton == "matrix_units":
            m = math.isqrt(bond_dim - 1) if bond_dim >= 1 else -1
            if m < 1 or 1 + m * m != bond_dim:
                raise ValueError(
                    f"skeleton='matrix_units' needs bond_dim = 1 + m^2 for "
                    f"an integer m >= 1 (got bond_dim={bond_dim}); valid "
                    f"values are 2, 5, 10, 17, 26, ..."
                )
            if m > d:
                raise ValueError(
                    f"skeleton='matrix_units' needs m <= d (got m={m} from "
                    f"bond_dim={bond_dim}, d={d})."
                )
        elif skeleton == "group":
            if group not in ("Zn", "Zn_product", "S3"):
                raise ValueError(
                    f"Unknown group {group!r} for skeleton='group'; "
                    f"expected 'Zn', 'Zn_product' or 'S3'."
                )
            if group == "Zn":
                group_size = group_n
            elif group == "Zn_product":
                group_size = int(np.prod(list(group_n)))
            else:
                group_size = 6  # |S3|
            if d != group_size or bond_dim != group_size:
                raise ValueError(
                    f"skeleton='group' needs d = bond_dim = |G| (group="
                    f"{group!r} has |G|={group_size}; got d={d}, "
                    f"bond_dim={bond_dim})."
                )
        else:
            raise ValueError(
                f"Unknown skeleton {skeleton!r}; expected 'cyclic', "
                f"'nested_flag', 'matrix_units' or 'group'."
            )

        def build_sites(n_sites_total: int, rng: np.random.Generator):
            if skeleton == "cyclic":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y = _cyclic_character_channels(d, bond_dim, r)
                    if self.verify:
                        _check_fusion_closure(y)
                        _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                    for g in range(bond_dim):
                        A[:, :, g, g] = y[g].T
                    return A

                l_vec = _flat_spectrum_boundary(bond_dim, rng)
                if self.verify:
                    _check_flat_spectrum(l_vec)

            elif skeleton == "nested_flag":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y = _nested_flag_channels(d, bond_dim, r)
                    if self.verify:
                        _check_nested_flag_closure(y)
                        _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                    for a in range(bond_dim):
                        A[:, :, a, a] = y[a].T
                    return A

                # Closed-form admissible boundary (flat torus), see docstring.
                l_vec = _nested_flag_boundary(bond_dim, rng)

            elif skeleton == "matrix_units":
                # Identity channel + m^2 matrix units; (B) is the unitarity
                # of Lambda + 1_m (see _matrix_unit_boundary).

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y, labels = _matrix_unit_channels(d, m, r)
                    if self.verify:
                        _check_matrix_unit_closure(y, labels)
                        if m < d:
                            # For m == d the identity equals sum_i y_(i,i), so
                            # the channels are linearly dependent by design.
                            _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                    for a in range(bond_dim):
                        A[:, :, a, a] = y[a].T
                    return A

                l_vec = _matrix_unit_boundary(m, rng)

            else:  # group
                # Group convolution: left-regular representation, one Haar
                # dressing W per site, Peter-Weyl boundary.
                elements, gmult, ginv, gid, irreps = _finite_group_table(group, group_n)
                if self.verify:
                    _check_group_representation(elements, gmult, irreps)
                gidx = {g: i for i, g in enumerate(elements)}
                y_group = _group_regular_channels(elements, gmult)
                gamma_group = _group_fusion_gamma(elements, gmult, ginv)

                def make_site(r: np.random.Generator) -> np.ndarray:
                    W = _haar_unitary(d, r)
                    y = np.einsum("ij,ajk,lk->ail", W, y_group, W.conj())
                    if self.verify:
                        _check_fusion_closure_general(y, gamma_group)
                        _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                    for a in range(bond_dim):
                        A[:, :, a, a] = y[a].T
                    return A

                l_vec = _group_fourier_boundary(elements, irreps, rng)
                if self.verify:
                    _check_group_boundary(l_vec, elements, gmult, gidx, gid)

            sites = self._sites(make_site, n_sites_total, rng)
            r_vec = np.ones(bond_dim, dtype=complex)
            return sites, l_vec, r_vec

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites)

    # -----------------------------------------------------------------
    # Mode II-B: bond-resolved fusion (skeletons in skeletons.py)
    # -----------------------------------------------------------------

    def fusion_bond_resolved(
        self,
        skeleton,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Mode II-B (bond-resolved fusion) from a fixed skeleton.

        The core tensors and boundary vectors of ``skeleton`` are fixed; only
        the physical dressing :math:`B_u\mapsto(W_n\otimes\bar W_n)B_u` is
        drawn per site (thesis, Mode II-B sampling remark).

        Parameters
        ----------
        skeleton : mpu_generators.skeletons.FusionSkeleton
            E.g. :func:`mpu_generators.skeletons.lee_yang_skeleton`.
        n_sites, n_blocks, blocking, max_blocking
            As in :meth:`branching`.

        Returns
        -------
        tensors, l_vec, r_vec
            As in :meth:`branching`.

        Notes
        -----
        Unlike the other modes, the sites are used in draw order (no
        reversal), so for per-site resampling the chains for increasing
        :math:`N` extend a common sequence at the *right* end.
        """
        if (n_sites is None) == (n_blocks is None):
            raise ValueError("Give exactly one of n_sites or n_blocks.")

        def _one_blocking(sites_L: int, L: int):
            if sites_L < 2:
                raise ValueError(f"n_sites must be >= 2, got {sites_L}.")
            if sites_L % L:
                raise ValueError(f"n_sites={sites_L} is not divisible by blocking={L}.")
            if sites_L // L < 2:
                raise ValueError(
                    f"n_sites={sites_L} with blocking={L} leaves "
                    f"{sites_L // L} tensor(s); need at least 2."
                )
            rng = np.random.default_rng(self.seed)
            sites = self._sites(lambda r: skeleton.dress(r), sites_L, rng)
            tensors = block_mpu_tensors(sites, L)
            l_vec, r_vec = skeleton.l_vec.copy(), skeleton.r_vec.copy()
            if self.verify:
                _verify_chain(tensors, l_vec, r_vec)
            return tensors, l_vec, r_vec

        return _run_blocking(
            _one_blocking, skeleton.d, skeleton.bond_dim, n_sites, n_blocks, blocking, max_blocking
        )

    # -----------------------------------------------------------------
    # Grafted composition (Mode I inside Mode II)
    # -----------------------------------------------------------------

    def grafted(
        self,
        d: int = 2,
        bond_dim: int = 2,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Grafted composition: a Mode I two-vertex path inside Mode II channels.

        Bond index :math:`(a, g)` with :math:`a\in\{0,1\}` a Mode I vertex
        (edges :math:`0\to0` identity, :math:`1\to\{1,0\}` resolved) and
        :math:`g` a cyclic channel; the site block is the operator
        :math:`K[a,a']\,y_g`, the boundary :math:`\ell_{(1,g)}=\ell_g`,
        :math:`\ell_{(0,g)}=0`, :math:`r=(1,\dots,1)`. The chain factorizes
        as :math:`U_N=K_N\,U_N^{\rm conv}`, a product of two independently
        certified unitaries (thesis ``thm:grafted-factorization``,
        ``rem:grafted-right-boundary``).

        Parameters
        ----------
        d, n_sites, n_blocks, blocking, max_blocking
            As in :meth:`branching`.
        bond_dim : int
            ``2 * n_channels`` with ``n_channels <= d``.

        Returns
        -------
        tensors, l_vec, r_vec
            As in :meth:`branching`.

        Notes
        -----
        The channels are the cyclic skeleton
        (:func:`_cyclic_character_channels`); the thesis states the theorem
        for the group-convolution (regular representation) skeleton, which
        coincides with it when ``n_channels`` divides ``d``. Grafting is not
        part of the tensor-first sampling paper.
        """
        if bond_dim % 2:
            raise ValueError(f"mode='grafted' has bond_dim = 2 * n_channels; got odd {bond_dim}.")
        n_channels = bond_dim // 2
        if n_channels > d:
            raise ValueError(
                f"mode='grafted' needs d >= bond_dim // 2 (got d={d}, "
                f"bond_dim={bond_dim} => n_channels={n_channels})."
            )
        k_edges = {0: [0], 1: [1, 0]}

        def build_sites(n_sites_total: int, rng: np.random.Generator):
            def make_site(r: np.random.Generator) -> np.ndarray:
                K = _mode1_site_tensor(2, k_edges, d, r, identity_rows=(0,))
                y = _cyclic_character_channels(d, n_channels, r)
                if self.verify:
                    _check_row_resolution(K)
                    _check_fusion_closure(y)
                    _check_channel_span_full_rank(y, n_channels)
                A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                for a in range(2):
                    for ap in range(2):
                        if not np.any(K[:, :, a, ap]):
                            continue
                        for g in range(n_channels):
                            # K stores the transposed Mode I block, y[g] is an
                            # operator (out, in): y[g].T @ K_store is the
                            # stored transpose of the operator K[a,ap] @ y[g].
                            A[:, :, a * n_channels + g, ap * n_channels + g] = (
                                y[g].T @ K[:, :, a, ap]
                            )
                return A

            l_conv = _flat_spectrum_boundary(n_channels, rng)
            if self.verify:
                _check_flat_spectrum(l_conv)
            sites = self._sites(make_site, n_sites_total, rng)
            l_vec = np.zeros(bond_dim, dtype=complex)
            l_vec[n_channels:] = l_conv
            r_vec = np.ones(bond_dim, dtype=complex)
            return sites, l_vec, r_vec

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites)


def generate_random_mpu(
    n_sites: int | None = None,
    d: int = 2,
    bond_dim: int = 2,
    mode: str = "branching",
    blocking: int | str = 1,
    uniform: bool = False,
    seed: int = 0,
    verify: bool = True,
    n_blocks: int | None = None,
    max_blocking: int = 8,
    **mode_kwargs,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Functional interface to the generators.

    Parameters
    ----------
    n_sites, d, bond_dim, blocking, n_blocks, max_blocking
        As in :meth:`MPUGenerator.branching`.
    mode : {"branching", "mode1", "fusion", "mode2", "grafted"}
        Which construction to use.
    uniform : bool
        Use :class:`~mpu_generators.uniform.UniformMPUGenerator`
        (translation invariant) instead of :class:`MPUGenerator`.
    seed, verify
        Passed to the generator constructor.
    **mode_kwargs
        Forwarded to the method, e.g. ``skeleton=`` for fusion or
        ``readout=`` for branching.

    Returns
    -------
    tensors, l_vec, r_vec
        As in :meth:`MPUGenerator.branching`.

    Examples
    --------
    >>> tensors, l, r = generate_random_mpu(n_sites=4, d=2, bond_dim=2, mode="fusion", seed=1)
    >>> len(tensors), tensors[0].shape
    (4, (2, 2, 2, 2))
    """
    from .uniform import UniformMPUGenerator

    cls = UniformMPUGenerator if uniform else MPUGenerator
    gen = cls(seed=seed, verify=verify)
    if mode in ("branching", "mode1"):
        return gen.branching(
            d=d,
            bond_dim=bond_dim,
            n_sites=n_sites,
            n_blocks=n_blocks,
            blocking=blocking,
            max_blocking=max_blocking,
            **mode_kwargs,
        )
    if mode in ("fusion", "mode2"):
        return gen.fusion(
            d=d,
            bond_dim=bond_dim,
            n_sites=n_sites,
            n_blocks=n_blocks,
            blocking=blocking,
            max_blocking=max_blocking,
            **mode_kwargs,
        )
    if mode == "grafted":
        return gen.grafted(
            d=d,
            bond_dim=bond_dim,
            n_sites=n_sites,
            n_blocks=n_blocks,
            blocking=blocking,
            max_blocking=max_blocking,
            **mode_kwargs,
        )
    raise ValueError(f"Unknown mode {mode!r}; expected 'branching', 'fusion' or 'grafted'.")
