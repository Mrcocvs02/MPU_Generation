# mpu_generators/base.py
r"""
General (non-uniform, per-site-resampled) certified random-MPU generator.

Implements Modes I, II, II-B and the Theorem-4 grafted composition from
"Certified Generators of Matrix Product Unitaries: A Minimal, Self-Contained
Account" [PDF-A]. Every mode confines the doubled transfer object $Q_n$ to a
fixed, $N$-independent family via an exact algebraic condition the sampling
recipe satisfies *by construction* -- see each method's docstring for the
condition and the recipe.

This module has one dependency direction only: it imports from
``mpu_decomposition`` (boundary-operator sweeps, canonicalization) and is
never imported *by* it, so it can be split into its own package later with
no rework. Assumption 1 (full-rank boundary operators) is a requirement of
:class:`~mpu_decomposition.MPU.GeneralMPU`'s *decomposition* algorithm, not
of certified generation itself -- the generator's own certificate is the
unitarity residual checked in ``_finalize_chain`` below. ``blocking="auto"``
is kept purely as an opt-in convenience for callers who *also* want
decomposable output, so its check is self-contained here rather than
imported from ``mpu_decomposition``.
"""
from __future__ import annotations

import math
from itertools import permutations, product

import numpy as np

from mpu_decomposition.utils import (
    block_mpu_tensors,
    compute_chain_boundary_operators,
)


def _satisfies_assumption_1(
    tensors: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-12,
) -> bool:
    """
    True iff every cut's boundary operators are full rank, i.e. iff
    :class:`~mpu_decomposition.MPU.GeneralMPU`'s minimal-bond-dimension
    check would accept this chain. Cheap: one left and one right sweep.
    Used only by ``blocking="auto"`` below -- not part of the generator's
    own unitarity certificate.
    """
    try:
        L2_list, R2_list, _ = compute_chain_boundary_operators(tensors, l_vec, r_vec)
    except ValueError:
        return False
    for L2, R2 in zip(L2_list, R2_list):
        if np.linalg.svd(L2, compute_uv=False).min() <= tol:
            return False
        if np.linalg.svd(R2, compute_uv=False).min() <= tol:
            return False
    return True


# =====================================================================
# Haar sampling
# =====================================================================


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed unitary: QR of a complex Ginibre matrix, R's phases divided out."""
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R)
    ph = ph / np.maximum(np.abs(ph), 1e-300)
    return (Q * ph.conj()[None, :]).astype(complex)


# =====================================================================
# Mode I: branching (row resolution)
# =====================================================================


def _row_resolution_blocks(
    d: int, targets: list[int], rng: np.random.Generator
) -> dict[int, np.ndarray]:
    r"""
    Sample one row of a Mode I tensor satisfying the row resolution condition

    .. math::
        \sum_u A[v,u]^\dagger A[v,u] = 1_d, \qquad
        A[v,u]^\dagger A[v,u'] = 0 \quad (u \neq u').                 \tag{RR}

    Draw ``V, W`` Haar, split ``1_d`` into orthogonal projectors ``P_i``
    onto consecutive column blocks of ``W`` (ranks as equal as possible),
    and set ``A[v, u_i] = V P_i W^\dagger``. Both halves of (RR) hold for
    *any* Haar draw ([PDF-A] Theorem 1's recipe), so no rejection sampling
    or per-sample repair is ever needed.

    ``targets`` need not be distinct branch labels in ``0..d-1`` specifically
    -- this is also reused, with ``d`` replaced by an enlarged physical
    dimension, for the appended combined-readout row (see
    :meth:`MPUGenerator.branching`'s ``readout`` argument).
    """
    m = len(targets)
    if not 1 <= m <= d:
        raise ValueError(
            f"A row needs 1 <= out-degree <= d outgoing edges; got {m} with d={d}."
        )
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


def _check_row_resolution(A_pdf: np.ndarray, atol: float = 1e-9) -> None:
    """Verify (RR) on every populated row of a Mode I tensor ``A[bL, bR, p_out, p_in]``."""
    D, _, d, _ = A_pdf.shape
    eye = np.eye(d, dtype=complex)
    for v in range(D):
        used = [u for u in range(D) if np.any(A_pdf[v, u])]
        if not used:
            continue
        acc = sum(A_pdf[v, u].conj().T @ A_pdf[v, u] for u in used)
        err = float(np.linalg.norm(acc - eye))
        if err > atol:
            raise RuntimeError(
                f"(RR) completeness failed at vertex {v}: "
                f"||sum_u A^dag A - 1|| = {err:.3e}"
            )
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                cross = float(
                    np.linalg.norm(A_pdf[v, used[i]].conj().T @ A_pdf[v, used[j]])
                )
                if cross > atol:
                    raise RuntimeError(
                        f"(RR) orthogonality failed at vertex {v}: "
                        f"||A[{used[i]}]^dag A[{used[j]}]|| = {cross:.3e}"
                    )


def _mode1_site_tensor(
    D: int,
    edges: dict[int, list[int]],
    d: int,
    rng: np.random.Generator,
    identity_rows: tuple[int, ...] = (),
) -> np.ndarray:
    """One Mode I site tensor ``A[bond_L, bond_R, p_out, p_in]`` (reference convention)."""
    A = np.zeros((D, D, d, d), dtype=complex)
    for v, targets in edges.items():
        if not targets:
            continue
        if v in identity_rows:
            if targets != [v]:
                raise ValueError(
                    f"identity_rows entry {v} must carry exactly one self-loop."
                )
            A[v, v] = np.eye(d, dtype=complex)
            continue
        for u, block in _row_resolution_blocks(d, targets, rng).items():
            A[v, u] = block
    return A


def _random_out_regular_digraph(
    D: int, d: int, rng: np.random.Generator, out_degree: int | None = None
) -> dict[int, list[int]]:
    """
    Random digraph on ``D`` vertices with a fixed out-degree, capped at ``d``
    because (RR) resolves at most ``d`` orthogonal branches per row. Self-loops
    and cycles are allowed -- Theorem 1 requires no acyclicity whatsoever.
    """
    m = min(d, D) if out_degree is None else min(out_degree, d, D)
    return {
        v: sorted(int(t) for t in rng.choice(D, size=m, replace=False))
        for v in range(D)
    }


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
    vertices at physical dimension ``d`` ([PDF-A] Remark 2, ``D_aux > d``
    case): a $d$-ary tree with ``D_aux`` leaves, built by repeatedly
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
    r"""
    Build the ``m`` extra site tensors realizing combined readout over
    ``readout`` at physical dimension ``d`` ([PDF-A] Remark 2), for any
    ``D_aux = len(readout)`` -- not just ``D_aux <= d`` (a single site,
    the ``m = 1`` case here is byte-for-byte the same computation the
    single-site construction used to do directly, verified by construction
    below, not merely claimed).

    Each level ``k -> k+1`` of :func:`_combined_readout_levels` becomes one
    site: bond ``L`` = number of groups at level ``k``, bond ``R`` = number
    of groups at level ``k + 1`` (or ``bond_dim`` at the last level, since
    that is what must plug into the rest of the digraph chain -- the
    populated columns are exactly ``readout``, everything else zero, same
    as the old single-site readout tensor). Every row is an independent
    (RR)-satisfying block from :func:`_row_resolution_blocks` -- Theorem
    1's own multi-site composition guarantee (already relied on by the
    rest of this module for the main digraph chain) is what certifies the
    *whole* cascade, not a new argument specific to this construction.

    Returned in level order, ``sites[0]`` having ``bond_L = 1`` (pairs
    with a length-1 ``l_pdf``, exactly like the old single readout site) --
    callers must append ``list(reversed(sites))`` to ``sites_pdf`` (PDF
    order), not ``sites`` itself, since :func:`_finalize_chain`'s global
    reversal is what restores this order in the final package-convention
    chain (verified directly against a dense dense_chain contraction, not
    assumed from the index bookkeeping alone).
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
        A = np.zeros((bond_L, bond_R, d, d), dtype=complex)
        for p, children in enumerate(child_ranges):
            blocks = _row_resolution_blocks(d, children, rng)
            for c, block in blocks.items():
                A[p, nxt[c][0] if k == m - 1 else c] = block
        sites.append(A)
    return sites


def _check_combined_readout_cascade(sites: list[np.ndarray], atol: float = 1e-9) -> None:
    """Verify (RR) at every row of every combined-readout cascade site (completeness + orthogonality)."""
    for A in sites:
        bond_L, bond_R, d, _ = A.shape
        for p in range(bond_L):
            used = [c for c in range(bond_R) if np.any(A[p, c])]
            if not used:
                continue
            acc = sum(A[p, c].conj().T @ A[p, c] for c in used)
            err = float(np.linalg.norm(acc - np.eye(d)))
            if err > atol:
                raise RuntimeError(
                    f"(RR) completeness failed in combined-readout cascade "
                    f"at row {p}: {err:.3e}"
                )
            for i in range(len(used)):
                for j in range(i + 1, len(used)):
                    cross = float(
                        np.linalg.norm(A[p, used[i]].conj().T @ A[p, used[j]])
                    )
                    if cross > atol:
                        raise RuntimeError(
                            f"(RR) orthogonality failed in combined-readout "
                            f"cascade at row {p}: {cross:.3e}"
                        )


# =====================================================================
# Mode II: fusion closure -- three named skeletons
# =====================================================================


def _group_convolution_channels(
    d: int, n_channels: int, rng: np.random.Generator
) -> np.ndarray:
    r"""
    Cyclic-group ($Z_M$) skeleton: ``y[g] = W diag(omega^{0g},...,omega^{(d-1)g}) W^dagger``.

    Monomial structure constants ``gamma^w_{uv} = delta_{w,(v-u) mod M}``.
    Returns an array of shape ``(n_channels, d, d)``.
    """
    omega = np.exp(2j * np.pi / n_channels)
    ks = np.arange(d)
    W = _haar_unitary(d, rng)
    return np.stack(
        [W @ np.diag(omega ** (ks * g)) @ W.conj().T for g in range(n_channels)]
    )


def _nested_flag_channels(d: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""
    Nested-flag skeleton [PDF-A] §3: ``y_a = W[:, a:] W[:, a:]^dagger`` (projector
    onto the last ``d - a`` columns of a Haar ``W``), closing via
    ``y_u^dagger y_v = y_{max(u,v)}`` -- monomial, ``gamma^w_{uv} = delta_{w,max(u,v)}``.
    This is the mechanism behind MCZ / diagonal-phase ([PDF-A] Table 6).

    Requires ``n_channels <= d`` (channel index ``a`` runs ``0..n_channels-1``,
    needs ``d - a >= 0`` columns to remain, i.e. ``a <= d``, and the top
    channel ``a = 0`` must be the full identity for ``y_0 = 1_d``).
    """
    if n_channels > d:
        raise ValueError(
            f"nested-flag skeleton needs n_channels <= d (got n_channels="
            f"{n_channels}, d={d})."
        )
    W = _haar_unitary(d, rng)
    return np.stack(
        [W[:, a:] @ W[:, a:].conj().T for a in range(n_channels)]
    )


def _matrix_unit_labels(m: int) -> list[tuple[int, int] | None]:
    r"""
    Ordered channel labels for the matrix-units skeleton: index ``0`` is a
    SEPARATE, distinct identity channel (label ``None``), followed by the
    ``m^2`` matrix-unit pairs ``(i, j)``, ``i, j = 0..m-1``.

    Earlier versions of this skeleton treated the identity as ``sum_i
    y_{(i,i)}`` (no separate channel, target ``e_{(p,q)} = delta_pq``) and
    found -- correctly, for *that* construction -- that no boundary
    restores unitarity beyond ``N=1`` (``sum_i y_{(i,i)} (x) y_{(i,i)} !=
    1_d (x) 1_d`` at ``N=2``). The fix, confirmed by an independent
    from-scratch re-derivation and direct ``dense_chain`` unitarity checks
    up to ``N=5`` (see ``CLAUDE.md``, "CORREZIONE CONFERMATA"), is that
    [PDF-A]'s skeleton always carried this separate identity channel --
    dropping it during implementation was the bug, not the skeleton
    itself.
    """
    return [None] + [(i, j) for i in range(m) for j in range(m)]


def _matrix_unit_channels(
    d: int, m: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int] | None]]:
    r"""
    Matrix-units channels [PDF-A] §3: a SEPARATE identity channel
    ``y_0 = 1_d`` plus ``y_{(i,j)} = w_i w_j^dagger`` from the first ``m``
    columns of a Haar ``W``. Satisfies (FC) (verified exactly by
    :func:`_check_matrix_unit_closure`):

    .. math::
        y_0^\dagger y_v = y_v, \quad y_u^\dagger y_0 = y_{(j,i)}
        \ (u=(i,j)), \quad y_0^\dagger y_0 = y_0, \quad
        y_{(i,j)}^\dagger y_{(k,l)} = \delta_{ik}\, y_{(j,l)}.

    Returns ``(y, labels)`` with ``y[0] = 1_d`` and ``labels`` from
    :func:`_matrix_unit_labels`. Requires ``m <= d``.
    """
    if m > d:
        raise ValueError(
            f"matrix-units skeleton needs m <= d (got m={m}, d={d})."
        )
    W = _haar_unitary(d, rng)
    w = [W[:, [i]] for i in range(m)]
    labels = _matrix_unit_labels(m)
    y = np.stack(
        [np.eye(d, dtype=complex)]
        + [w[i] @ w[j].conj().T for (i, j) in labels[1:]]
    )
    return y, labels


def _matrix_unit_boundary(m: int, rng: np.random.Generator) -> np.ndarray:
    r"""
    Solve (B) for the matrix-units skeleton (identity channel + ``m^2``
    matrix-unit channels), target ``e_0 = 1``, ``e_{(p,q)} = 0`` for every
    pair. Writing ``Lambda_{ij} := l_{(i,j)}``, (B) reduces to ``Lambda +
    Lambda^dagger + Lambda^dagger Lambda = 0``, i.e. ``V := Lambda + 1_m``
    must be unitary -- so ``l_{(i,j)} = (V - 1_m)_{ij}`` for an arbitrary
    ``V in U(m)``, ``l_0 = 1``. See ``CLAUDE.md``'s "CORREZIONE CONFERMATA"
    section for the from-scratch derivation of this equation.
    """
    labels = _matrix_unit_labels(m)
    V = _haar_unitary(m, rng)
    l_pdf = np.zeros(len(labels), dtype=complex)
    l_pdf[0] = 1.0
    for idx, lab in enumerate(labels):
        if lab is not None:
            i, j = lab
            l_pdf[idx] = V[i, j] - (1.0 if i == j else 0.0)
    return l_pdf


def _check_fusion_closure_general(
    y: np.ndarray, gamma: np.ndarray, atol: float = 1e-9
) -> None:
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
                    f"(FC) failed at (u={u}, v={v}): "
                    f"||y[u]^dag y[v] - y[(v-u) mod M]|| = {err:.3e}"
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
                    f"(FC) failed [matrix-units] at ({a_lab},{b_lab}): "
                    f"||...|| = {err:.3e}"
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
    r"""
    Solve (B) for the $Z_M$ cyclic skeleton: with monomial constants
    ``gamma^w_{uv} = delta_{w,(v-u) mod M}``, (B) reduces to a flat Fourier
    spectrum ``|DFT(l)[k]| = 1``. Sampling random phases and
    inverse-transforming lands on the solution variety generically (every
    channel populated).
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
    return np.fft.ifft(np.exp(1j * theta)).astype(complex)


def _check_flat_spectrum(l_vec: np.ndarray, atol: float = 1e-9) -> None:
    """Verify (B) for the cyclic skeleton: ``|DFT(l)[k]| = 1`` for every k."""
    err = float(np.max(np.abs(np.abs(np.fft.fft(l_vec)) - 1.0)))
    if err > atol:
        raise RuntimeError(f"(B) failed: max_k | |DFT(l)[k]| - 1 | = {err:.3e}")


def _solve_boundary_general(
    gamma: np.ndarray, e: np.ndarray, rng: np.random.Generator, n_restarts: int = 8
) -> np.ndarray:
    r"""
    Solve (B), ``sum_{u,v} conj(l_u) l_v gamma^w_{uv} = e_w`` for every ``w``,
    for an arbitrary skeleton. This is a system that is linear in the outer
    product ``L[u,v] := conj(l_u) l_v`` (a rank-1, PSD, trace-``|l|^2``
    Hermitian matrix) but not directly in ``l``, so it is solved by: (1)
    solve the linear system in ``L`` (a generically unique rank-1 PSD
    solution, since the physical construction only ever needs ``one``
    valid boundary, not the general small-dimensional case handled by the
    named closed forms), (2) recover ``l`` from ``L``'s dominant eigenvector.

    Falls back across ``n_restarts`` random small perturbations of the
    least-squares solve if the naive solve is not close enough to rank 1
    (the skeletons used here -- nested-flag, matrix-units -- are verified
    to admit an exact rank-1 solution via the named closed forms in
    :mod:`tests`, so this general solver exists for skeletons without a
    hand-derived closed form).
    """
    n = gamma.shape[0]
    # Flatten gamma^w_{uv} into a (n, n*n) matrix acting on vec(L).
    Gamma = gamma.reshape(n, n * n)
    best = None
    for _ in range(max(1, n_restarts)):
        L_flat, *_ = np.linalg.lstsq(Gamma, e, rcond=None)
        L = L_flat.reshape(n, n)
        L = 0.5 * (L + L.conj().T)  # project onto Hermitian
        w, V = np.linalg.eigh(L)
        idx = int(np.argmax(w))
        if w[idx] <= 0:
            continue
        l_candidate = V[:, idx] * np.sqrt(w[idx])
        resid = float(
            np.max(
                np.abs(
                    np.einsum("u,v,wuv->w", l_candidate.conj(), l_candidate, gamma)
                    - e
                )
            )
        )
        if best is None or resid < best[0]:
            best = (resid, l_candidate)
    if best is None or best[0] > 1e-6:
        raise RuntimeError(
            "Could not find a boundary vector solving (B) for this skeleton "
            f"(best residual {None if best is None else best[0]:.3e})."
        )
    return best[1].astype(complex)


def _finite_group_table(group: str, n: int | list[int] = 0):
    r"""
    Elements, multiplication, inverse and identity for a small finite
    group, plus its irreducible representations (as lists of matrices, one
    per element, same order as ``elements``) -- built independently from
    :meth:`mpu_generators.uniform.UniformMPUGenerator.group_automaton`'s
    own table builder (that one is Mode I and already tested; duplicating
    the ~10 lines here avoids touching working code for an unrelated Mode
    II mechanism).

    ``group``:
      - ``"Zn"``: cyclic group of order ``n`` (abelian). Irreps: the ``n``
        one-dimensional characters $\rho_k(g) = \omega^{kg}$,
        $\omega = e^{2\pi i/n}$ -- standard, works for any ``n``.
      - ``"Zn_product"``: direct product $Z_{n_1} \times \cdots \times
        Z_{n_k}$ (``n`` is a list of factor orders here, not a single
        int) -- *any* finite abelian group, by the fundamental theorem of
        finite abelian groups (every finite abelian group is a product of
        cyclic factors). Task 6: generalizes ``"Zn"`` (the single-factor
        case) the same way a multi-dimensional DFT generalizes a 1-D one
        -- irreps are still all 1-dimensional (abelian groups only ever
        have 1-dim irreps), now one PER COMBINATION of per-factor
        characters, $\rho_{\vec k}(\vec g) = \prod_i \omega_i^{k_i g_i}$.
      - ``"S3"``: the symmetric group on 3 letters (dihedral of order 6,
        the smallest non-abelian group). ``n`` is ignored. Irreps: trivial
        (1-dim), sign (1-dim, +1 on the identity/two 3-cycles, -1 on the
        three transpositions), and the 2-dim "standard" representation
        (S3 as the symmetries of a triangle: 3-cycles are rotations by
        120 degrees, transpositions are reflections) -- hand-derived and
        verified (see :func:`_check_group_representation`), not looked up
        from a representation-theory library. Only ``S3`` is supported
        here (not general ``Sn``): finding every irrep of ``Sn`` for
        general ``n`` needs Specht-module-level machinery this module
        deliberately avoids (see Task 1's scope note in ``CLAUDE.md``).

    Returns ``(elements, mult, inv, identity, irreps)``, ``irreps`` a list
    of ``(dim, [rho(g) for g in elements])`` pairs.
    """
    if group == "Zn_product":
        factors = list(n)
        if len(factors) < 1 or any(f < 1 for f in factors):
            raise ValueError(f"group='Zn_product' needs a non-empty list of factors >= 1, got {factors}.")
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
            (1, [np.array([[omega ** (k * g)]], dtype=complex) for g in elements])
            for k in range(n)
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
    r"""
    Verify every claimed irrep is actually a unitary representation:
    ``rho(g) rho(h) == rho(g*h)`` and ``rho(g)`` unitary, for every
    ``g, h``. Direct, independent check -- never trust a hand-derived
    representation without re-verifying the homomorphism property itself.
    """
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
    r"""
    Regular-representation Mode II channels for a finite group $G$
    (abelian or not): physical dimension $d=|G|$, one channel per group
    element, $y_g|h\rangle = |gh\rangle$ (left multiplication as a
    permutation matrix). Elementary group theory -- true for *any* finite
    group, not just abelian ones -- gives monomial fusion closure exactly:
    $y_g^\dagger y_h = y_{g^{-1}h}$ (verified by
    :func:`_check_fusion_closure_general` against the ``gamma`` this
    function's companion :func:`_group_fusion_gamma` builds).
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
    r"""
    Solve (B) for :func:`_group_regular_channels` via the non-abelian
    generalization of the cyclic skeleton's flat-Fourier-spectrum trick
    (:func:`_flat_spectrum_boundary`): the generic linear solver
    (:func:`_solve_boundary_general`) does not reliably find a solution
    for this skeleton (verified: residual ~0.75, not close to 0, even for
    an *abelian* test group where a solution is known to exist) -- this is
    a from-scratch derivation instead, independently re-verified below by
    :func:`_check_group_boundary`'s brute-force double sum, not merely
    trusted from the Fourier-transform algebra alone.

    For a finite group, every function $f:G\to\mathbb C$ has a Fourier
    transform at each irrep $\rho$, $\hat f(\rho) = \sum_g f(g)\rho(g)$,
    invertible as $f(g) = \frac1{|G|}\sum_\rho d_\rho
    \operatorname{Tr}(\rho(g)^\dagger \hat f(\rho))$ (Peter-Weyl). Writing
    out (B), $\sum_g \overline{l(g)}\,l(gw) = \delta_{w,e}$, as a Fourier
    transform in $w$ shows the left side's transform at $\rho$ equals
    $\hat l(\rho)^\dagger \hat l(\rho)$, while $\delta_e$'s transform at
    $\rho$ is $\rho(e) = 1_{d_\rho}$ -- so (B) holds *iff* $\hat l(\rho)$
    is unitary at *every* irrep (exactly "flat spectrum", generalized from
    1x1 characters to $d_\rho\times d_\rho$ blocks). Sampling a fresh Haar
    unitary $U_\rho$ per irrep and inverse-transforming lands on the
    solution variety generically, mirroring
    :func:`_flat_spectrum_boundary`'s own random-phase-then-IDFT recipe.
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
    G = len(elements)
    for wi, w in enumerate(elements):
        target = 1.0 if w == identity else 0.0
        S = 0.0 + 0.0j
        for gi, g in enumerate(elements):
            S += np.conj(l_vec[gi]) * l_vec[idx[mult(g, w)]]
        err = abs(S - target)
        if err > atol:
            raise RuntimeError(
                f"(B) failed [group] at w={w}: |S(w) - target| = {err:.3e}"
            )


# =====================================================================
# Common tail: convention conversion, blocking, chain verification
# =====================================================================


def _pdf_to_package_tensor(A_pdf: np.ndarray) -> np.ndarray:
    """``A[bond_L, bond_R, p_out, p_in]`` -> this package's ``A[p_in, p_out, bond_L, bond_R]``."""
    return np.ascontiguousarray(A_pdf.transpose(3, 2, 0, 1))


def _finalize_chain(
    sites_pdf: list[np.ndarray],
    l_pdf: np.ndarray,
    r_pdf: np.ndarray,
    blocking: int,
    verify: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Shared tail for every mode: reverse to package site order, convert
    convention, block, and (optionally) verify the assembled chain's global
    unitarity residual.
    """
    tensors = [_pdf_to_package_tensor(A) for A in reversed(sites_pdf)]
    tensors = block_mpu_tensors(tensors, blocking)
    l_vec = l_pdf.copy()
    r_vec = r_pdf.copy()
    if verify:
        _, _, residual = compute_chain_boundary_operators(tensors, l_vec, r_vec)
        err = abs(1.0 - complex(residual))
        if err > 1e-8:
            raise RuntimeError(
                f"Generated MPU is not unitary: chain residual {residual} "
                f"deviates from 1 by {err:.3e}."
            )
    return tensors, l_vec, r_vec


# =====================================================================
# MPUGenerator
# =====================================================================


class MPUGenerator:
    """
    General (non-uniform) certified random-MPU generator: every mode
    resamples a fresh Haar dressing at every physical site, per [PDF-A].

    :class:`UniformMPUGenerator` overrides exactly the resample-vs-reuse
    decision (:meth:`_sites`) to get the translationally-invariant variant
    of every mode below "for free", without duplicating any mode's
    algebra -- see its docstring.
    """

    #: If False (only in :class:`UniformMPUGenerator`), every mode draws its
    #: per-site dressing once and reuses it at every site instead of
    #: resampling -- the single override point for translational invariance.
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
        build_sites_pdf,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Shared validation + auto-blocking search, mirroring the original
        ``generate_random_mpu``/``_generate_random_mpu_fixed_blocking``.
        ``build_sites_pdf(n_sites_total, rng)`` must return
        ``(sites_pdf, l_pdf, r_pdf)`` in PDF/reference convention.
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
            sites_pdf, l_pdf, r_pdf = build_sites_pdf(sites_L, rng)
            return _finalize_chain(sites_pdf, l_pdf, r_pdf, L, self.verify)

        if blocking == "auto":
            attempts = []
            for L in range(1, max_blocking + 1):
                sites_L = n_blocks * L if n_blocks is not None else n_sites
                if n_blocks is None and (n_sites % L or n_sites // L < 2):
                    continue
                try:
                    cand = _one_blocking(sites_L, L)
                except ValueError as exc:
                    attempts.append(f"L={L}: {exc}")
                    continue
                if _satisfies_assumption_1(*cand):
                    return cand
                attempts.append(f"L={L}: rank-deficient boundary operators")
            raise ValueError(
                f"No blocking <= {max_blocking} satisfies Assumption 1. Tried: "
                + "; ".join(attempts)
            )

        L = int(blocking)
        sites_L = n_blocks * L if n_blocks is not None else n_sites
        return _one_blocking(sites_L, L)

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
        r"""
        Mode I / row resolution ([PDF-A] §2). Site tensors indexed by a
        random digraph on ``bond_dim`` vertices, every row satisfying (RR)
        (see :func:`_row_resolution_blocks`). Each branch ``U^{(v)}_N`` is
        individually unitary (Theorem 1).

        ``readout``: if ``None`` (default), single-vertex readout,
        ``l = e_0`` -- the ``D_aux = 1`` degenerate case of [PDF-A]'s
        combined-readout corollary (Remark 1). If given, a list of vertices
        ``V_read``: appends ``m = max(1, ceil(log_d(D_aux)))`` extra
        physical-``d`` sites realizing the combined readout over
        ``D_aux = len(readout)`` vertices ([PDF-A] Remark 1's corollary
        for ``D_aux <= d``, generalized to any ``D_aux`` via Remark 2's
        padded multi-site construction for ``D_aux > d`` --
        :func:`_combined_readout_sites`, a $d$-ary tree of independent
        (RR)-satisfying rows, never truncating: every one of the ``D_aux``
        vertices gets an exact branch, padding only ever means "resolve in
        more than one physical measurement," never "drop a branch").
        ``D_aux <= d`` is the ``m = 1`` case of the exact same
        construction, not a separate code path.
        """
        def build_sites_pdf(n_sites_total: int, rng: np.random.Generator):
            edges = _random_out_regular_digraph(bond_dim, d, rng)

            def make_site(r: np.random.Generator) -> np.ndarray:
                site_edges = edges if not self._resample else _random_out_regular_digraph(
                    bond_dim, d, r
                )
                return _mode1_site_tensor(bond_dim, site_edges, d, r)

            sites_pdf = self._sites(make_site, n_sites_total, rng)

            if self.verify:
                for A_pdf in sites_pdf[: (1 if not self._resample else n_sites_total)]:
                    _check_row_resolution(A_pdf)

            if readout is None:
                l_pdf = np.zeros(bond_dim, dtype=complex)
                l_pdf[0] = 1.0
            else:
                cascade_sites = _combined_readout_sites(readout, d, bond_dim, rng)
                if self.verify:
                    _check_combined_readout_cascade(cascade_sites)
                # cascade_sites[0] has bond_L=1 (pairs with l_pdf below);
                # appended in REVERSE level order so _finalize_chain's
                # global reversal restores [site_0, site_1, ...] order in
                # the final package-convention chain -- see
                # _combined_readout_sites' docstring, verified directly
                # against dense_chain (not assumed from the bookkeeping).
                sites_pdf = sites_pdf + list(reversed(cascade_sites))
                l_pdf = np.array([1.0], dtype=complex)

            r_pdf = np.ones(bond_dim, dtype=complex)
            return sites_pdf, l_pdf, r_pdf

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites_pdf)

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
        r"""
        Mode II / fusion closure ([PDF-A] §3). Tensor strictly diagonal in
        the virtual index, ``A[a,b] = delta_ab y_a``, closing under (FC);
        the boundary solves the single global equation (B).

        ``skeleton``:
            - ``"cyclic"`` (default): $Z_M$ characters
              (:func:`_group_convolution_channels`), monomial closure,
              boundary via the flat-Fourier-spectrum trick
              (:func:`_flat_spectrum_boundary`). Needs ``bond_dim <= d``.
            - ``"nested_flag"``: MCZ/diagonal-phase's mechanism
              (:func:`_nested_flag_channels`). Needs ``bond_dim <= d``.
            - ``"matrix_units"``: subspace-rotation's mechanism -- a
              SEPARATE identity channel plus ``m^2`` matrix-unit channels
              ``y_{(i,j)} = w_i w_j^dagger`` (:func:`_matrix_unit_channels`,
              :func:`_matrix_unit_boundary`). ``bond_dim`` must equal
              ``1 + m^2`` for an integer ``m <= d`` (i.e. one of
              ``2, 5, 10, 17, 26, ...``); earlier revisions of this method
              dropped the separate identity channel and, correctly for
              *that* construction, could not restore unitarity past
              ``N=1`` -- see ``CLAUDE.md``'s "CORREZIONE CONFERMATA" for
              the from-scratch re-derivation that found and fixed the gap.
            - ``"group"``: regular representation of a finite group $G$,
              abelian or not (:func:`_group_regular_channels`) -- one
              channel per group element, $y_g|h\rangle=|gh\rangle$;
              ``d = bond_dim = |G|`` exactly. ``"cyclic"`` is really the
              $G=Z_M$ special case of this, restricted to abelian groups
              because it diagonalizes the channels via 1-dim characters;
              this option also covers non-abelian groups (``group="S3"``,
              the smallest non-abelian group) via a genuinely different
              boundary construction (:func:`_group_fourier_boundary`) that
              does not require the channels to commute, AND any other
              finite abelian group (``group="Zn_product"``, ``group_n`` a
              list of cyclic factor orders -- Task 6: every finite abelian
              group is a product of cyclic factors, by the fundamental
              theorem, so this covers all of them, not just single-factor
              $Z_M$). See :func:`_finite_group_table` for the
              ``group``/``group_n`` options.
        """
        if skeleton == "cyclic":
            if bond_dim > d:
                raise ValueError(
                    f"skeleton='cyclic' needs d >= bond_dim (got d={d}, "
                    f"bond_dim={bond_dim})."
                )
        elif skeleton == "nested_flag":
            if bond_dim > d:
                raise ValueError(
                    f"skeleton='nested_flag' needs d >= bond_dim (got d={d}, "
                    f"bond_dim={bond_dim})."
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

        def build_sites_pdf(n_sites_total: int, rng: np.random.Generator):
            if skeleton == "cyclic":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y = _group_convolution_channels(d, bond_dim, r)
                    if self.verify:
                        _check_fusion_closure(y)
                        _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((bond_dim, bond_dim, d, d), dtype=complex)
                    for g in range(bond_dim):
                        A[g, g] = y[g]
                    return A

                l_pdf = _flat_spectrum_boundary(bond_dim, rng)
                if self.verify:
                    _check_flat_spectrum(l_pdf)

            elif skeleton == "nested_flag":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y = _nested_flag_channels(d, bond_dim, r)
                    if self.verify:
                        _check_nested_flag_closure(y)
                        _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((bond_dim, bond_dim, d, d), dtype=complex)
                    for a in range(bond_dim):
                        A[a, a] = y[a]
                    return A

                # (B) for gamma^w_{uv} = delta_{w, max(u,v)}: channel 0 is the
                # identity (y_0 = 1_d, full flag), so e_w = delta_{w,0}, and
                # gamma^0_{uv} = delta_{u,0} delta_{v,0} -- the only way to hit
                # w=0 is (u,v)=(0,0), since max(u,v)=0 forces both zero. So (B)
                # at w=0 gives |l_0|^2 = 1; at w>0, sum_{u,v: max(u,v)=w}
                # conj(l_u) l_v = 0. This is solved exactly by any l with
                # |l_0|=1 and l_1=...=l_{bond_dim-1}=0 (trivial channel-0-only
                # solution) OR by the richer per-instance solve below, which
                # searches for a genuinely populated solution when one exists
                # (matches MCZ's own two-discrete-point structure at bond_dim=2).
                gamma = np.zeros((bond_dim, bond_dim, bond_dim))
                for u in range(bond_dim):
                    for v in range(bond_dim):
                        gamma[max(u, v), u, v] = 1.0
                e = np.zeros(bond_dim)
                e[0] = 1.0
                l_pdf = _solve_boundary_general(gamma, e, rng)

            elif skeleton == "matrix_units":
                # CORRECTED construction (see CLAUDE.md "CORREZIONE
                # CONFERMATA"): a SEPARATE identity channel y_0 = 1_d is
                # required alongside the m^2 matrix-unit channels -- an
                # earlier revision used e_{(p,q)} = delta_pq with NO
                # separate channel and, correctly for that construction,
                # found no boundary restores unitarity past N=1 (that
                # diagnosis is still valid for the construction it tested;
                # it just wasn't [PDF-A]'s actual skeleton). With the
                # separate channel, (B) reduces to Lambda + Lambda^dagger +
                # Lambda^dagger Lambda = 0 (:func:`_matrix_unit_boundary`),
                # verified here to give N=1..5 unitarity at machine
                # precision across many (d, m) shapes.

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y, labels = _matrix_unit_channels(d, m, r)
                    if self.verify:
                        _check_matrix_unit_closure(y, labels)
                        if m < d:
                            # At m == d the separate identity channel is
                            # necessarily a linear combination of the m^2
                            # matrix-unit channels (sum_i y_(i,i) = 1_d
                            # exactly when {w_i} is a full basis), so the
                            # channel set is rank-deficient by construction
                            # -- not a bug, just m == d's redundancy; skip
                            # the full-rank check only in that case.
                            _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((bond_dim, bond_dim, d, d), dtype=complex)
                    for a in range(bond_dim):
                        A[a, a] = y[a]
                    return A

                l_pdf = _matrix_unit_boundary(m, rng)

            else:  # group
                # Regular representation, any finite group (Task 1: the
                # non-abelian S3 case is the whole point -- "cyclic" only
                # ever covers abelian groups because it diagonalizes via
                # 1-dim characters). The generic _solve_boundary_general
                # solver does NOT work here (verified: residual ~0.75, no
                # solution found, even for an abelian test group where one
                # is known to exist) -- _group_fourier_boundary is a
                # from-scratch, independently-verified replacement (see
                # its docstring and CLAUDE.md's Task 1 writeup).
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
                    A = np.zeros((bond_dim, bond_dim, d, d), dtype=complex)
                    for a in range(bond_dim):
                        A[a, a] = y[a]
                    return A

                l_pdf = _group_fourier_boundary(elements, irreps, rng)
                if self.verify:
                    _check_group_boundary(l_pdf, elements, gmult, gidx, gid)

            sites_pdf = self._sites(make_site, n_sites_total, rng)
            r_pdf = np.ones(bond_dim, dtype=complex)
            return sites_pdf, l_pdf, r_pdf

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites_pdf)

    # -----------------------------------------------------------------
    # Mode II-B: bond-resolved fusion (see skeletons.py for the solver)
    # -----------------------------------------------------------------

    def fusion_bond_resolved(
        self,
        skeleton,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""
        Mode II-B / bond-resolved fusion ([PDF-A] §4). ``skeleton`` must be
        a :class:`mpu_generators.skeletons.FusionSkeleton` (built once, e.g.
        via :func:`mpu_generators.skeletons.lee_yang_skeleton`) -- the core
        tensors ``B_u`` and boundary ``l`` are fixed once by solving the
        Intertwiner condition (I) and (B); only the per-site Haar dressing
        ``B_u -> (W_n (x) conj(W_n)) B_u`` is resampled here.

        Unlike the other three modes, :meth:`FusionSkeleton.dress` already
        returns tensors in this package's own convention (it is built
        directly from an existing package-convention construction, not the
        PDF/reference one the other modes start from -- see its docstring),
        so this method does its own finalize step rather than going through
        :meth:`_generate`/``_finalize_chain``'s PDF-convention reversal.
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
                _, _, residual = compute_chain_boundary_operators(tensors, l_vec, r_vec)
                err = abs(1.0 - complex(residual))
                if err > 1e-8:
                    raise RuntimeError(
                        f"Generated MPU is not unitary: chain residual "
                        f"{residual} deviates from 1 by {err:.3e}."
                    )
            return tensors, l_vec, r_vec

        if blocking == "auto":
            attempts = []
            for L in range(1, max_blocking + 1):
                sites_L = n_blocks * L if n_blocks is not None else n_sites
                if n_blocks is None and (n_sites % L or n_sites // L < 2):
                    continue
                try:
                    cand = _one_blocking(sites_L, L)
                except ValueError as exc:
                    attempts.append(f"L={L}: {exc}")
                    continue
                if _satisfies_assumption_1(*cand):
                    return cand
                attempts.append(f"L={L}: rank-deficient boundary operators")
            raise ValueError(
                f"No blocking <= {max_blocking} satisfies Assumption 1. Tried: "
                + "; ".join(attempts)
            )

        L = int(blocking)
        sites_L = n_blocks * L if n_blocks is not None else n_sites
        return _one_blocking(sites_L, L)

    # -----------------------------------------------------------------
    # Theorem 4: grafted composition
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
        r"""
        Theorem 4 grafted composition ([PDF-A] §5): a Mode-I two-vertex path
        algebra $K_N$ grafted inside the channels of a Mode-II cyclic
        convolution skeleton, sharing one complementary pair across all
        channels. $U_N = K_N \cdot U_N^{conv}$, a product of two
        independently certified unitaries. Requires even ``bond_dim``
        (``= 2 * n_channels``).
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

        def build_sites_pdf(n_sites_total: int, rng: np.random.Generator):
            def make_site(r: np.random.Generator) -> np.ndarray:
                K = _mode1_site_tensor(2, k_edges, d, r, identity_rows=(0,))
                y = _group_convolution_channels(d, n_channels, r)
                if self.verify:
                    _check_row_resolution(K)
                    _check_fusion_closure(y)
                    _check_channel_span_full_rank(y, n_channels)
                A = np.zeros((bond_dim, bond_dim, d, d), dtype=complex)
                for a in range(2):
                    for ap in range(2):
                        if not np.any(K[a, ap]):
                            continue
                        for g in range(n_channels):
                            A[a * n_channels + g, ap * n_channels + g] = K[a, ap] @ y[g]
                return A

            l_conv = _flat_spectrum_boundary(n_channels, rng)
            if self.verify:
                _check_flat_spectrum(l_conv)
            sites_pdf = self._sites(make_site, n_sites_total, rng)
            l_pdf = np.zeros(bond_dim, dtype=complex)
            l_pdf[n_channels:] = l_conv
            r_pdf = np.ones(bond_dim, dtype=complex)
            return sites_pdf, l_pdf, r_pdf

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites_pdf)


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
    """
    Back-compat / convenience entry point matching the original function's
    signature (``mode in {"branching","mode1","fusion","mode2","grafted"}``),
    dispatching to :class:`MPUGenerator` / :class:`UniformMPUGenerator`.
    Extra ``**mode_kwargs`` (e.g. ``skeleton=`` for fusion, ``readout=`` for
    branching) are forwarded to the corresponding method.
    """
    from .uniform import UniformMPUGenerator

    cls = UniformMPUGenerator if uniform else MPUGenerator
    gen = cls(seed=seed, verify=verify)
    if mode in ("branching", "mode1"):
        return gen.branching(
            d=d, bond_dim=bond_dim, n_sites=n_sites, n_blocks=n_blocks,
            blocking=blocking, max_blocking=max_blocking, **mode_kwargs,
        )
    if mode in ("fusion", "mode2"):
        return gen.fusion(
            d=d, bond_dim=bond_dim, n_sites=n_sites, n_blocks=n_blocks,
            blocking=blocking, max_blocking=max_blocking, **mode_kwargs,
        )
    if mode == "grafted":
        return gen.grafted(
            d=d, bond_dim=bond_dim, n_sites=n_sites, n_blocks=n_blocks,
            blocking=blocking, max_blocking=max_blocking, **mode_kwargs,
        )
    raise ValueError(f"Unknown mode {mode!r}; expected 'branching', 'fusion' or 'grafted'.")
