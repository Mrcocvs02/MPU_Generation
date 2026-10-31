# mpu_generators/base.py
r"""Random generators of open-boundary Matrix Product Unitaries (MPUs).

Every generator returns site tensors ``A[p_in, p_out, bond_L, bond_R]`` and
boundary vectors ``(l, r)`` whose contraction

.. math::
    U_N = \sum_{v_0,\dots,v_N} \ell_{v_0} r_{v_N}\,
          A[v_0,v_1]\otimes\cdots\otimes A[v_{N-1},v_N]

is unitary for every chain length :math:`N`. For Modes I and II and the
grafted composition this follows from an algebraic identity of the local
tensor and boundary data (thesis ``thm:mode1-unitarity``,
``thm:mode2-unitarity``, ``thm:grafted-factorization``), not from tests at a
few lengths; for Mode II-B it is validated numerically only.

Constructions
-------------
* **Mode I** (row resolution, RR): :meth:`MPUGenerator.branching`; every row
  of the bond digraph resolves the identity into orthogonal branches.
* **Mode II** (fusion closure, FC): :meth:`MPUGenerator.fusion`; bond-diagonal
  channels closing under a fusion rule, boundary (B) on an admissible manifold.
* **Mode II-B**: :meth:`MPUGenerator.fusion_bond_resolved`; fixed cores of a
  :class:`~mpu_generators.skeletons.FusionSkeleton`, resampled dressing.
* **Grafted**: :meth:`MPUGenerator.grafted`; a Mode I path inside the cyclic
  Mode II channels.

Certificates and checks
-----------------------
With ``verify=True`` the certificate of each construction is re-checked
numerically at an absolute tolerance of ``1e-9`` on every sampled tensor
(once for the repeated tensor in the uniform case); this table is the single
reference for the per-mode checks. "Independence" is the linear independence of
the channels :math:`y_u`, skipped for ``matrix_units`` when ``m == d`` and for
``cyclic`` when ``bond_dim > d`` (where it fails by design).

==============================  ==========================================================
Construction                    Checks run with ``verify=True``
==============================  ==========================================================
Mode I (``branching``)         (RR) on every row, also of the readout cascade
``fusion``, ``cyclic``          (FC); (B) flat Fourier spectrum; independence (if
                                ``bond_dim <= d``)
``fusion``, ``group``           irrep representation property; (FC); (B) by the direct
                                double sum; independence
``fusion``, ``nested_flag``     (FC); independence. (B) is not checked: the boundary is
                                constructed on the admissible manifold
``fusion``, ``matrix_units``    (FC); independence (if ``m < d``). (B) is not checked, as
                                for ``nested_flag``
Grafted                         (RR) for the Mode I factor; (FC), (B) (flat spectrum) and
                                independence for the channels
Mode II-B                       none (the intertwiner helpers of
                                :mod:`mpu_generators.skeletons` are not called)
==============================  ==========================================================

Independently, every assembled chain passes the numerical safeguards of
:func:`_finalize_chain` (:func:`_verify_chain` for Mode II-B): the
normalization :math:`\operatorname{Tr}(U^\dagger U)/d^N=1` for any :math:`N`,
and the dense condition :math:`U^\dagger U=1`, to tolerance, when
:math:`d^N\le 2^{10}`.

Conventions
-----------
Tensors are ``A[p_in, p_out, bond_L, bond_R]`` (first physical index = input,
as in :mod:`TreeMPU`). The block stored at ``A[:, :, v, u]`` is the
*transpose* of the operator :math:`A[v,u]` (rows = output) used in the thesis.
Boundary vectors are contracted without complex conjugation. Sites are built
in construction order and reversed by :func:`_finalize_chain`. The package
depends on :mod:`TreeMPU` only for :func:`~TreeMPU.utils.block_mpu_tensors`,
:func:`~TreeMPU.utils.compute_chain_boundary_operators`,
:func:`~TreeMPU.utils.satisfies_assumption_1` and the dense reference in
:mod:`TreeMPU.checks`.

References
----------
[Styliaris et al., PRL 135, 260602]; M. Vassallo, master's thesis
(``thm:mode1-unitarity``, ``thm:mode2-unitarity``,
``thm:grafted-factorization``) and "Tensor-First Sampling of Exact
Matrix-Product Unitaries" (in preparation).

Examples
--------
>>> from mpu_generators import MPUGenerator
>>> tensors, l_vec, r_vec = MPUGenerator(seed=0).branching(d=2, bond_dim=2, n_sites=6)
>>> len(tensors), tensors[0].shape
(6, (2, 2, 2, 2))
"""

from __future__ import annotations

import math
import warnings
from itertools import permutations, product

import numpy as np
from TreeMPU.utils import (
    block_mpu_tensors,
    compute_chain_boundary_operators,
)
from TreeMPU.utils import satisfies_assumption_1 as _satisfies_assumption_1

#: Largest blocked tensor (d**L, d**L, D, D) in complex128 that blocking may
#: build (512 MiB). The cost grows exponentially in L, so the size is checked
#: before allocation for explicit and automatic blocking.
_AUTO_BLOCKING_MAX_BYTES = 512 * 1024 * 1024

#: Default absolute tolerance of the ``_check_*`` certificates (Frobenius norm
#: or max-norm of the residual); tight, because every certified construction
#: holds to machine precision, yet far above double-precision round-off.
_CHECK_ATOL = 1e-9
#: Floor on ``|diag R|`` in the QR phase fix of the Haar sampler; it only
#: prevents a division by zero (the diagonal is nonzero almost surely).
_PHASE_FLOOR = 1e-300


# --- Haar sampling ---


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a Haar-distributed unitary of size ``dim``.

    QR decomposition of a complex Ginibre matrix :math:`Z=QR`, with the phases
    of :math:`\mathrm{diag}(R)` absorbed into :math:`Q` so that the result is
    Haar distributed (Mezzadri, Notices AMS 54, 592 (2007)). Consumes exactly
    two ``standard_normal((dim, dim))`` draws from ``rng``.

    Parameters
    ----------
    dim : int
        Matrix size.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    ndarray
        Complex unitary of shape ``(dim, dim)``.
    """
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R)
    ph = ph / np.maximum(np.abs(ph), _PHASE_FLOOR)
    return (Q * ph[None, :]).astype(complex)


# --- Mode I: branching (row resolution) ---


def _row_resolution_blocks(
    d: int, targets: list[int], rng: np.random.Generator, ranks: list[int] | None = None
) -> dict[int, np.ndarray]:
    r"""Sample one row of a Mode I tensor obeying row resolution (RR).

    .. math::
        \sum_u A[v,u]^\dagger A[v,u] = 1_d, \qquad
        A[v,u]^\dagger A[v,u'] = 0 \quad (u \neq u'),              \tag{RR}

    together with the mirror identities for :math:`A[v,u]A[v,u']^\dagger`
    (thesis ``thm:mode1-unitarity``; paper, "row-resolution identity" in
    the section on exact certification mechanisms).

    Sampler: draw :math:`V, W` Haar in :math:`U(d)`, let :math:`P_i` be the
    orthogonal projector onto the :math:`i`-th consecutive block of columns
    of :math:`W` (block sizes ``ranks``; by default as equal as possible), and
    set :math:`A[v,u_i] = V P_i W^\dagger`. Both halves of (RR) hold for every
    draw and every rank partition, so no rejection is needed.

    Parameters
    ----------
    d : int
        Physical dimension (or the enlarged dimension of a readout site).
    targets : list of int
        Target vertices :math:`u_i` of the outgoing edges of this row.
    rng : numpy.random.Generator
        Consumes two Haar draws (:math:`V` then :math:`W`).
    ranks : list of int, optional
        Rank of each block, positive and summing to ``d`` (one per target).
        Default: the balanced partition.

    Returns
    -------
    dict[int, ndarray]
        ``{u_i: A[v, u_i]}`` as :math:`d\times d` matrices acting
        output <- input.

    Raises
    ------
    ValueError
        If the number of targets is not in ``1..d``, or ``ranks`` does not
        consist of ``len(targets)`` positive integers summing to ``d``.

    Notes
    -----
    The same unitary :math:`W` supplies both the resolved subspaces and the
    input frame, so the joint law of the blocks differs from the one obtained
    with independent :math:`W_1 P_i W_2^\dagger`; both satisfy (RR).
    """
    m = len(targets)
    if not 1 <= m <= d:
        raise ValueError(f"A row needs 1 <= out-degree <= d outgoing edges; got {m} with d={d}.")
    if ranks is None:
        base, rem = divmod(d, m)
        ranks = [base + (1 if i < rem else 0) for i in range(m)]
    elif len(ranks) != m or any(int(k) < 1 for k in ranks) or sum(ranks) != d:
        raise ValueError(f"ranks {list(ranks)} must be {m} positive integers summing to d={d}.")

    V = _haar_unitary(d, rng)
    W = _haar_unitary(d, rng)

    blocks: dict[int, np.ndarray] = {}
    offset = 0
    for i, u in enumerate(targets):
        cols = W[:, offset : offset + ranks[i]]
        blocks[u] = V @ (cols @ cols.conj().T) @ W.conj().T
        offset += ranks[i]
    return blocks


def _check_row_resolution(A: np.ndarray, atol: float = _CHECK_ATOL) -> None:
    r"""Check (RR) on every populated row of ``A[p_in, p_out, bond_L, bond_R]``.

    The stored block is the transpose of the operator,
    ``A[:, :, v, u] = A_op[v, u].T``, so
    :math:`\sum_u A_{\rm op}^\dagger A_{\rm op} = (\sum_u B B^\dagger)^\top`
    with ``B = A[:, :, v, u]``; the residuals computed below are therefore
    those of (RR) itself. A row is populated if some block ``A[:, :, v, u]``
    is nonzero; empty rows are skipped.

    Parameters
    ----------
    A : ndarray
        Tensor of shape ``(d, d, D, D)``.
    atol : float
        Absolute Frobenius-norm tolerance of each residual.

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
    ranks: str | dict[int, list[int]] = "balanced",
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
    ranks : "balanced", "random" or dict
        Rank partition of every row: balanced (default), a uniformly random
        composition of ``d`` into ``len(edges[v])`` positive parts (drawn
        from ``rng`` just before the row's Haar unitaries), or explicit
        ``{v: [rank of edge to edges[v][0], ...]}``. Not applied to
        ``identity_rows``.

    Returns
    -------
    ndarray
        Shape ``(d, d, D, D)``; entry ``[:, :, v, u]`` stores the transposed
        operator :math:`A[v,u]^\top` (index order ``p_in, p_out``).

    Raises
    ------
    ValueError
        If an ``identity_rows`` vertex does not have exactly the single
        target ``[v]``, or ``ranks`` is not one of the accepted forms.
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
        if ranks == "balanced":
            row_ranks = None
        elif ranks == "random":
            row_ranks = _random_composition(d, len(targets), rng)
        elif isinstance(ranks, dict):
            row_ranks = list(ranks[v])
        else:
            raise ValueError(f"ranks must be 'balanced', 'random' or a dict; got {ranks!r}.")
        for u, block in _row_resolution_blocks(d, targets, rng, row_ranks).items():
            A[:, :, v, u] = block.T
    return A


def _random_composition(n: int, k: int, rng: np.random.Generator) -> list[int]:
    """Return a uniformly random composition of ``n`` into ``k`` positive parts.

    The ``k - 1`` cut points are chosen without replacement from ``1..n-1``
    (stars and bars).

    Parameters
    ----------
    n : int
        Total to be split.
    k : int
        Number of parts, ``1 <= k <= n``.
    rng : numpy.random.Generator
        Consumes one ``choice`` draw if ``k > 1``, none otherwise.

    Returns
    -------
    list of int
        ``k`` positive integers summing to ``n``.
    """
    cuts = np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False)) if k > 1 else []
    edges = [0, *(int(c) for c in cuts), n]
    return [edges[i + 1] - edges[i] for i in range(k)]


def _check_digraph(edges: dict[int, list[int]], D: int, d: int) -> dict[int, list[int]]:
    """Validate a user-supplied Mode I digraph and return a normalized copy.

    Every vertex ``0..D-1`` needs between 1 and ``d`` distinct targets in
    ``0..D-1``: a row without outgoing edge would break the telescoping sum
    of the unitarity proof, and (RR) resolves at most ``d`` orthogonal
    branches. Parallel edges are excluded because two projectors on the same
    tensor entry would add to a single, non-projective block.

    Parameters
    ----------
    edges : dict[int, list[int]]
        ``{v: targets}`` as supplied by the user.
    D : int
        Number of vertices (bond dimension).
    d : int
        Physical dimension; the maximal out-degree.

    Returns
    -------
    dict[int, list[int]]
        ``{v: [targets]}`` with plain ``int`` entries.

    Raises
    ------
    ValueError
        If the vertex set is not ``0..D-1``, or a vertex violates the
        out-degree or distinctness conditions.
    """
    if sorted(edges) != list(range(D)):
        raise ValueError(f"digraph must have exactly the vertices 0..{D - 1}; got {sorted(edges)}.")
    out = {}
    for v in range(D):
        t = [int(u) for u in edges[v]]
        if not 1 <= len(t) <= d:
            raise ValueError(f"vertex {v} has out-degree {len(t)}; need 1 <= out-degree <= d={d}.")
        if len(set(t)) != len(t) or any(not 0 <= u < D for u in t):
            raise ValueError(f"vertex {v}: targets {t} must be distinct vertices in 0..{D - 1}.")
        out[v] = t
    return out


def _check_ranks(ranks: dict[int, list[int]], edges: dict[int, list[int]], d: int) -> None:
    """Validate an explicit rank partition against its digraph.

    For every vertex ``v`` of ``edges``, ``ranks[v]`` must hold one positive
    integer per target of ``v`` and sum to ``d``.

    Parameters
    ----------
    ranks : dict[int, list[int]]
        ``{v: [rank of the edge to edges[v][0], ...]}``.
    edges : dict[int, list[int]]
        Validated digraph ``{v: targets}`` (see :func:`_check_digraph`).
    d : int
        Physical dimension; the required sum of every row's ranks.

    Raises
    ------
    ValueError
        If an entry is missing or malformed.
    """
    for v, targets in edges.items():
        if v not in ranks:
            raise ValueError(f"ranks has no entry for vertex {v}.")
        r = [int(k) for k in ranks[v]]
        if len(r) != len(targets) or any(k < 1 for k in r) or sum(r) != d:
            raise ValueError(
                f"ranks[{v}] = {r} must be {len(targets)} positive integers summing to d={d}."
            )


def _random_out_regular_digraph(
    D: int, d: int, rng: np.random.Generator, out_degree: int | str | None = None
) -> dict[int, list[int]]:
    """Return a random digraph ``{v: sorted targets}`` on ``D`` vertices.

    Every vertex gets ``m = min(int(out_degree), min(d, D))`` distinct
    targets (``m = min(d, D)`` if ``out_degree`` is ``None``), chosen
    uniformly without replacement (self-loops allowed). With
    ``out_degree="random"`` each vertex first draws its own ``m`` uniformly
    in ``1..min(d, D)``. The cap at ``d`` is forced by (RR): a row resolves
    at most ``d`` mutually orthogonal branches. Cycles are allowed; the
    Mode I certificate needs no acyclicity.

    Parameters
    ----------
    D : int
        Number of vertices.
    d : int
        Physical dimension.
    rng : numpy.random.Generator
        Consumed vertex by vertex (in the ``"random"`` case, one degree draw
        before each target draw).
    out_degree : int, "random" or None
        Fixed out-degree (capped at ``min(d, D)``), ``"random"``, or ``None``
        for the cap.

    Returns
    -------
    dict[int, list[int]]
        ``{v: sorted targets}`` with plain ``int`` entries.

    Raises
    ------
    ValueError
        If ``out_degree`` is an integer below 1 (``0`` does not default to
        the cap).
    """
    cap = min(d, D)
    if out_degree == "random":
        out = {}
        for v in range(D):
            m = int(rng.integers(1, cap + 1))
            out[v] = sorted(int(t) for t in rng.choice(D, size=m, replace=False))
        return out
    m = cap if out_degree is None else min(int(out_degree), cap)
    if m < 1:
        raise ValueError(f"out_degree must be >= 1, got {out_degree}.")
    return {v: sorted(int(t) for t in rng.choice(D, size=m, replace=False)) for v in range(D)}


def _even_split(items: list, k: int) -> list[list]:
    """Split ``items`` into ``k`` contiguous groups whose sizes differ by at most 1.

    The order of ``items`` is preserved; the first ``len(items) % k`` groups
    are the larger ones.

    Parameters
    ----------
    items : list
        Sequence to split.
    k : int
        Number of groups, ``k >= 1``.

    Returns
    -------
    list of list
        ``k`` consecutive slices of ``items`` (some empty if
        ``k > len(items)``).
    """
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
    r"""Return the group hierarchy of a combined readout cascade.

    Builds a :math:`d`-ary tree with ``len(readout)`` leaves (thesis
    ``rem:cascaded-readout``, case :math:`D_{\rm aux}>d`): starting from the
    single group ``list(readout)``, every group of size > 1 is split into up
    to ``d`` contiguous, nearly equal subgroups, and a group of size 1 passes
    through unchanged (its row is a trivial out-degree-1 (RR) row). Splitting
    preserves the order of ``readout``, so the last level is exactly
    ``[[readout[0]], ..., [readout[-1]]]`` and no leaf needs padding.

    Parameters
    ----------
    readout : list of int
        Vertices to be read out.
    d : int
        Physical dimension; the maximal fan-out of each split.

    Returns
    -------
    list of list of list of int
        ``levels`` with ``levels[0] == [list(readout)]`` and ``levels[-1]``
        all singletons. ``len(levels) - 1`` is the number of extra physical
        sites, :math:`\max(1,\lceil\log_d D_{\rm aux}\rceil)`; the minimum
        of one site also covers ``len(readout) == 1``.
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
    r"""Build the extra sites of a combined readout over ``readout``.

    Each level ``k -> k+1`` of :func:`_combined_readout_levels` becomes one
    site with ``bond_L`` = number of groups at level ``k`` and ``bond_R`` =
    number of groups at level ``k + 1``. At the last level ``bond_R`` is
    ``bond_dim``, to plug into the bulk chain; the populated columns are
    exactly ``readout`` and all others vanish. Every row is an independent
    (RR) row from :func:`_row_resolution_blocks`, so the cascade is certified
    by the Mode I theorem like the bulk.

    Parameters
    ----------
    readout : list of int
        Vertices to be read out (valid indices of ``range(bond_dim)``).
    d : int
        Physical dimension.
    bond_dim : int
        Bond dimension of the bulk chain.
    rng : numpy.random.Generator
        Consumed level by level, row by row.

    Returns
    -------
    list of ndarray
        Sites ``(d, d, bond_L, bond_R)`` in level order; ``sites[0]`` has
        ``bond_L = 1`` (pairs with :math:`\ell=(1)`). Callers append
        ``list(reversed(sites))`` because :func:`_finalize_chain` reverses the
        whole list.
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


def _check_combined_readout_cascade(sites: list[np.ndarray], atol: float = _CHECK_ATOL) -> None:
    """Check (RR) on every row of every combined-readout cascade site.

    Same completeness and cross-orthogonality residuals as
    :func:`_check_row_resolution`, applied to each tensor of ``sites``
    (shape ``(d, d, bond_L, bond_R)``, ``p_in`` first).

    Parameters
    ----------
    sites : list of ndarray
        Cascade tensors from :func:`_combined_readout_sites`.
    atol : float
        Absolute Frobenius-norm tolerance of each residual.

    Raises
    ------
    RuntimeError
        If a residual exceeds ``atol`` (Frobenius norm).
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


# --- Mode II: fusion closure -- three named skeletons ---


def _cyclic_character_channels(d: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""Build the cyclic skeleton of :math:`\mathbb Z_M` characters.

    The channels are
    :math:`y_g = W\,\mathrm{diag}(\omega^{0\cdot g},\dots,\omega^{(d-1)g})\,W^\dagger`,
    where :math:`\omega=e^{2\pi i/M}`, :math:`M` = ``n_channels``,
    :math:`g=0,\dots,M-1`, and :math:`W` is one Haar unitary. The channels
    form a unitary representation of :math:`\mathbb Z_M` and close
    monomially, :math:`y_u^\dagger y_v = y_{(v-u)\bmod M}`.

    Parameters
    ----------
    d : int
        Physical dimension.
    n_channels : int
        Number of channels :math:`M`.
    rng : numpy.random.Generator
        Consumes one Haar draw.

    Returns
    -------
    ndarray
        Shape ``(n_channels, d, d)``, operators acting output <- input.

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


#: Alias of :func:`_cyclic_character_channels`.
_group_convolution_channels = _cyclic_character_channels


def _nested_flag_channels(d: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    r"""Build the nested-flag skeleton of shrinking projectors.

    The channel :math:`y_a = P_a` projects onto the last :math:`d-a` columns
    of a Haar unitary :math:`W`, for :math:`a = 0,\dots,M-1` with
    :math:`M` = ``n_channels``. Hence :math:`y_0 = 1_d`
    and :math:`\operatorname{rank} y_a = d-a`. The flag is nested
    (:math:`P_{a}\ge P_{a+1}`) and closes monomially,
    :math:`y_u^\dagger y_v = y_{\max(u,v)}`. For :math:`d=2` with a fixed
    basis this is the MCZ / diagonal-phase mechanism (thesis
    ``sec:beyond-qca``).

    Parameters
    ----------
    d : int
        Physical dimension.
    n_channels : int
        Number of channels :math:`M`, at most ``d``.
    rng : numpy.random.Generator
        Consumes one Haar draw (after the size check).

    Returns
    -------
    ndarray
        Shape ``(n_channels, d, d)``.

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
    r"""Return the channel labels of the matrix-units skeleton.

    Index 0 is a separate identity channel (label ``None``), followed by the
    :math:`m^2` matrix units :math:`(i,j)`, :math:`i,j=0,\dots,m-1`, in
    row-major order. The identity must be a channel of its own: if it were
    represented as :math:`\sum_i y_{(i,i)}`, then
    :math:`\sum_i y_{(i,i)}^{\otimes 2}\neq 1_d^{\otimes 2}` and no boundary
    would give a unitary for :math:`N\ge 2`.

    Parameters
    ----------
    m : int
        Size of the matrix-unit basis, ``m >= 1``.

    Returns
    -------
    list
        ``1 + m**2`` labels: ``None`` followed by the pairs ``(i, j)``.
    """
    return [None] + [(i, j) for i in range(m) for j in range(m)]


def _matrix_unit_channels(
    d: int, m: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int] | None]]:
    r"""Build the matrix-units channels, :math:`y_{(i,j)}=w_iw_j^\dagger`.

    :math:`w_i` are the first :math:`m` columns of one Haar :math:`W` (one
    Haar draw from ``rng``). The channels satisfy (FC), checked by
    :func:`_check_matrix_unit_closure`:

    .. math::
        y_0^\dagger y_v = y_v, \quad y_u^\dagger y_0 = y_{(j,i)}
        \ (u=(i,j)), \quad y_0^\dagger y_0 = y_0, \quad
        y_{(i,j)}^\dagger y_{(k,l)} = \delta_{ik}\, y_{(j,l)}.

    Parameters
    ----------
    d : int
        Physical dimension.
    m : int
        Number of matrix-unit vectors, ``m <= d``.
    rng : numpy.random.Generator
        Consumes one Haar draw (after the size check).

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
    r"""Sample a boundary vector on the matrix-units admissible manifold.

    With target :math:`e_0=1`, :math:`e_{(p,q)}=0` and
    :math:`\Lambda_{ij}:=\ell_{(i,j)}`, the boundary condition (B) reduces to
    :math:`\Lambda+\Lambda^\dagger+\Lambda^\dagger\Lambda=0`, i.e.
    :math:`V:=\Lambda+1_m` is unitary. The manifold is therefore
    :math:`\ell_0=1`, :math:`\ell_{(i,j)}=(V-1_m)_{ij}`, :math:`V\in U(m)`, and
    :math:`V` is drawn from the Haar measure. (B) holds by construction and is
    not re-checked at generation.

    Parameters
    ----------
    m : int
        Size of the matrix-unit basis.
    rng : numpy.random.Generator
        Consumes one Haar draw of :math:`U(m)`.

    Returns
    -------
    ndarray
        Complex array of shape ``(1 + m**2,)`` in the order of
        :func:`_matrix_unit_labels`.
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


def _check_fusion_closure_general(
    y: np.ndarray, gamma: np.ndarray, atol: float = _CHECK_ATOL
) -> None:
    r"""Check (FC) for a skeleton with general structure constants ``gamma``.

    The condition is ``y[u]^dagger y[v] == sum_w gamma[w,u,v] y[w]`` for all
    ``u, v``.

    Parameters
    ----------
    y : ndarray
        Channels, shape ``(n_channels, d, d)``.
    gamma : ndarray
        Structure constants, shape ``(n_channels, n_channels, n_channels)``,
        indexed ``[w, u, v]``.
    atol : float
        Absolute Frobenius-norm tolerance.

    Raises
    ------
    RuntimeError
        If the identity fails for some ``(u, v)``.
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


def _check_fusion_closure(y: np.ndarray, atol: float = _CHECK_ATOL) -> None:
    """Check (FC) for the cyclic skeleton: ``y[u]^dag y[v] == y[(v-u) mod M]``.

    Parameters
    ----------
    y : ndarray
        Channels, shape ``(M, d, d)``.
    atol : float
        Absolute Frobenius-norm tolerance.

    Raises
    ------
    RuntimeError
        If the identity fails by more than ``atol`` (Frobenius norm).
    """
    M = y.shape[0]
    for u in range(M):
        for v in range(M):
            err = float(np.linalg.norm(y[u].conj().T @ y[v] - y[(v - u) % M]))
            if err > atol:
                raise RuntimeError(
                    f"(FC) failed at (u={u}, v={v}): ||y[u]^dag y[v] - y[(v-u) mod M]|| = {err:.3e}"
                )


def _check_nested_flag_closure(y: np.ndarray, atol: float = _CHECK_ATOL) -> None:
    """Check (FC) for the nested-flag skeleton: ``y[u]^dag y[v] == y[max(u,v)]``.

    Parameters
    ----------
    y : ndarray
        Channels, shape ``(n_channels, d, d)``.
    atol : float
        Absolute Frobenius-norm tolerance.

    Raises
    ------
    RuntimeError
        If the identity fails by more than ``atol`` (Frobenius norm).
    """
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
    y: np.ndarray, labels: list[tuple[int, int] | None], atol: float = _CHECK_ATOL
) -> None:
    r"""Check (FC) for the matrix-units skeleton with identity channel.

    The identity channel has label ``None`` at index 0. Verified for all
    pairs: ``y_0^dag y_v = y_v``, ``y_u^dag y_0 = y_{(j,i)}`` for
    ``u=(i,j)``, ``y_0^dag y_0 = y_0`` and
    ``y_{(i,j)}^dag y_{(k,l)} = delta_ik y_{(j,l)}``.

    Parameters
    ----------
    y : ndarray
        Channels, shape ``(1 + m**2, d, d)``.
    labels : list
        Channel labels from :func:`_matrix_unit_labels`.
    atol : float
        Absolute Frobenius-norm tolerance.

    Raises
    ------
    RuntimeError
        If an identity fails by more than ``atol`` (Frobenius norm).
    """
    index = {lab: idx for idx, lab in enumerate(labels)}

    def expected(a_lab, b_lab):
        # Index of the channel equal to y[a_lab]^dag y[b_lab], or None if the
        # product vanishes.
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


def _check_channel_span_full_rank(y: np.ndarray, n_channels: int, tol: float = _CHECK_ATOL) -> None:
    """Check that the ``n_channels`` channels are linearly independent in ``M_d(C)``.

    Linear independence ensures that the requested bond dimension is not
    reduced by the MPU canonical form.

    Parameters
    ----------
    y : ndarray
        Channels, shape ``(n_channels, d, d)``.
    n_channels : int
        Number of channels (first axis of ``y``).
    tol : float
        Singular-value tolerance passed to ``numpy.linalg.matrix_rank``.

    Raises
    ------
    RuntimeError
        If the rank of the flattened channels (tolerance ``tol``) is below
        ``n_channels``.
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
    r"""Sample a boundary vector for the cyclic skeleton (flat Fourier spectrum).

    With :math:`\gamma^w_{uv}=\delta_{w,(v-u)\bmod M}` the boundary condition
    (B) is :math:`|\mathrm{DFT}(\ell)_k|=1` for all :math:`k`. Uniform random
    phases are drawn and inverse-transformed (uniform measure on the torus
    :math:`T^M`).

    Parameters
    ----------
    n_channels : int
        Number of channels :math:`M`.
    rng : numpy.random.Generator
        Consumes ``n_channels`` uniform draws.

    Returns
    -------
    ndarray
        Complex array of shape ``(n_channels,)``.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
    return np.fft.ifft(np.exp(1j * theta)).astype(complex)


def _check_flat_spectrum(l_vec: np.ndarray, atol: float = _CHECK_ATOL) -> None:
    """Check (B) for the cyclic skeleton: ``|DFT(l)[k]| = 1`` for every ``k``.

    Parameters
    ----------
    l_vec : ndarray
        Boundary vector, shape ``(M,)``.
    atol : float
        Absolute tolerance on ``| |DFT(l)[k]| - 1 |``.

    Raises
    ------
    RuntimeError
        If the largest deviation exceeds ``atol``.
    """
    err = float(np.max(np.abs(np.abs(np.fft.fft(l_vec)) - 1.0)))
    if err > atol:
        raise RuntimeError(f"(B) failed: max_k | |DFT(l)[k]| - 1 | = {err:.3e}")


def _nested_flag_boundary(bond_dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary vector on the nested-flag admissible manifold.

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
    gives the MCZ boundary :math:`\ell=(1,-2)`. (B) holds by construction and
    is not re-checked at generation.

    Parameters
    ----------
    bond_dim : int
        Number of channels :math:`\chi`.
    rng : numpy.random.Generator
        Consumes ``bond_dim - 1`` uniform angle draws.

    Returns
    -------
    ndarray
        Complex array of shape ``(bond_dim,)`` with ``l[0] = 1``.
    """
    l_vec = np.zeros(bond_dim, dtype=complex)
    l_vec[0] = 1.0 + 0.0j
    for w in range(1, bond_dim):
        S_w = np.sum(l_vec[:w])  # |S_w| = 1 by the recursion in the docstring
        theta = rng.uniform(0.0, 2.0 * np.pi)
        l_vec[w] = -S_w + abs(S_w) * np.exp(1j * theta)
    return l_vec


def _finite_group_table(group: str, n: int | list[int] = 0):
    r"""Return the multiplication table and irreps of a small finite group.

    Parameters
    ----------
    group : {"Zn", "Zn_product", "S3"}
        ``"Zn"``: cyclic group of order ``n``, with the ``n`` characters
        :math:`\rho_k(g)=\omega^{kg}`, :math:`\omega=e^{2\pi i/n}`.
        ``"Zn_product"``: direct product :math:`Z_{n_1}\times\cdots\times
        Z_{n_k}` of cyclic factors (any finite abelian group), with the
        products of per-factor characters as irreps. ``"S3"``: symmetric group
        on three letters; irreps are the trivial, the sign and the
        two-dimensional standard representation (``n`` is ignored).
    n : int or list of int
        Order of ``Zn``, or list of factor orders for ``Zn_product``.

    Returns
    -------
    elements : list
        Group elements (ints, tuples of ints, or permutation tuples).
    mult : callable
        ``mult(g, h)`` returns the product ``g h``.
    inv : callable
        ``inv(g)`` returns the inverse.
    identity
        The neutral element.
    irreps : list of tuple
        ``(dim, [rho(g) for g in elements])`` for each irrep, with ``rho(g)``
        a unitary ``(dim, dim)`` array.

    Raises
    ------
    ValueError
        If ``group`` is unknown, or ``n`` is invalid for the chosen group.

    Notes
    -----
    The tensor-first sampling paper states the group-convolution results for
    every finite group; this implementation provides the abelian groups and
    :math:`S_3` only, because general groups need their irreducible
    representations as input. The homomorphism property of every irrep is
    checked by :func:`_check_group_representation`. For :math:`S_3`,
    ``mult(g, h)`` is the composition ``g`` after ``h``; the standard
    representation maps the 3-cycle ``r`` to a rotation by 120 degrees and the
    transposition ``s`` to a reflection, with every element written as
    :math:`r^k` or :math:`s\,r^k`.
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
                """Return the 1x1 character ``prod_i omega_i**(k_i g_i)`` at ``g``."""
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
            """Return the sign of the permutation ``perm`` of ``range(3)``."""
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

        # Classify each element as r^k or s r^k; s r^k is reached by multiplying
        # r on the right, because s r^k = r^-k s in a dihedral group.
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


def _check_group_representation(elements, mult, irreps, atol: float = _CHECK_ATOL) -> None:
    r"""Check that every irrep is unitary and a homomorphism.

    The homomorphism condition is :math:`\rho(g)\rho(h)=\rho(gh)` for all
    ``g, h``.

    Parameters
    ----------
    elements : list
        Group elements, as returned by :func:`_finite_group_table`.
    mult : callable
        Group multiplication.
    irreps : list of tuple
        ``(dim, [rho(g) for g in elements])`` pairs.
    atol : float
        Absolute Frobenius-norm tolerance.

    Raises
    ------
    RuntimeError
        If an irrep is not unitary or not a homomorphism.
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
    r"""Build the channels of the left-regular representation of :math:`G`.

    Physical dimension :math:`d=|G|`, one channel per element,
    :math:`y_g|h\rangle=|gh\rangle`. Closure is monomial for every finite
    group, :math:`y_g^\dagger y_h = y_{g^{-1}h}` (structure constants from
    :func:`_group_fusion_gamma`).

    Parameters
    ----------
    elements : list
        Group elements, as returned by :func:`_finite_group_table`.
    mult : callable
        ``mult(g, h)`` returns the product ``g h``.

    Returns
    -------
    ndarray
        Shape ``(|G|, |G|, |G|)``, operators acting output <- input.
    """
    D = len(elements)
    idx = {g: i for i, g in enumerate(elements)}
    y = np.zeros((D, D, D), dtype=complex)
    for gi, g in enumerate(elements):
        for hi, h in enumerate(elements):
            y[gi, idx[mult(g, h)], hi] = 1.0
    return y


def _group_fusion_gamma(elements, mult, inv) -> np.ndarray:
    r"""Return the structure constants of :func:`_group_regular_channels`.

    The array ``gamma[w, u, v]`` of shape ``(|G|, |G|, |G|)`` equals
    :math:`\delta_{w, g_u^{-1} g_v}`.

    Parameters
    ----------
    elements : list
        Group elements, as returned by :func:`_finite_group_table`.
    mult : callable
        ``mult(g, h)`` returns the product ``g h``.
    inv : callable
        ``inv(g)`` returns the inverse.

    Returns
    -------
    ndarray
        Real array ``gamma[w, u, v]``.
    """
    D = len(elements)
    idx = {g: i for i, g in enumerate(elements)}
    gamma = np.zeros((D, D, D))
    for u in range(D):
        for v in range(D):
            w = idx[mult(inv(elements[u]), elements[v])]
            gamma[w, u, v] = 1.0
    return gamma


def _group_fourier_boundary(elements, irreps, rng: np.random.Generator) -> np.ndarray:
    r"""Sample a boundary vector on the Peter-Weyl admissible manifold.

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
    back. With ``verify=True`` the result is re-checked by the direct double
    sum in :func:`_check_group_boundary`.

    Parameters
    ----------
    elements : list
        Group elements, as returned by :func:`_finite_group_table`.
    irreps : list of tuple
        ``(dim, [rho(g) for g in elements])`` for each irrep.
    rng : numpy.random.Generator
        Consumes one Haar draw per irrep.

    Returns
    -------
    ndarray
        Complex array of shape ``(|G|,)``.
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
    l_vec: np.ndarray, elements, mult, idx, identity, atol: float = _CHECK_ATOL
) -> None:
    """Check (B) by the direct double sum, independent of the Fourier construction.

    Verifies ``sum_g conj(l(g)) l(gw) == delta_{w,identity}`` for every ``w``;
    ``idx`` maps elements to their positions in ``elements``.

    Parameters
    ----------
    l_vec : ndarray
        Boundary vector, shape ``(|G|,)``.
    elements : list
        Group elements, as returned by :func:`_finite_group_table`.
    mult : callable
        ``mult(g, h)`` returns the product ``g h``.
    idx : dict
        Position of each element in ``elements``.
    identity
        The neutral element.
    atol : float
        Absolute tolerance on ``|S(w) - delta_{w,identity}|``.

    Raises
    ------
    RuntimeError
        If the identity fails for some ``w`` by more than ``atol``.
    """
    for wi, w in enumerate(elements):
        target = 1.0 if w == identity else 0.0
        S = 0.0 + 0.0j
        for gi, g in enumerate(elements):
            S += np.conj(l_vec[gi]) * l_vec[idx[mult(g, w)]]
        err = abs(S - target)
        if err > atol:
            raise RuntimeError(f"(B) failed [group] at w={w}: |S(w) - target| = {err:.3e}")


# --- Common tail: site-order reversal, blocking, chain verification ---


def _finalize_chain(
    sites: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    blocking: int,
    verify: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    r"""Reverse, block and (optionally) verify a freshly built chain.

    Builders append sites in construction order, which is the reverse of the
    physical order; the list is reversed here and then grouped into blocks of
    ``blocking`` consecutive sites. Sites are already in the convention
    ``A[p_in, p_out, bond_L, bond_R]``.

    Parameters
    ----------
    sites : list of ndarray
        Site tensors in construction order.
    l_vec, r_vec : ndarray
        Boundary vectors (copied, never modified in place).
    blocking : int
        Number of physical sites merged into one tensor; must divide
        ``len(sites)``.
    verify : bool
        Run :func:`_verify_chain`.

    Returns
    -------
    tensors : list of ndarray
        Blocked tensors in physical left-to-right order.
    l_vec, r_vec : ndarray
        Copies of the boundary vectors.

    Raises
    ------
    ValueError
        If ``blocking < 1`` or does not divide ``len(sites)``
        (from :func:`~TreeMPU.utils.block_mpu_tensors`).
    RuntimeError
        If ``verify`` is true and a safeguard of :func:`_verify_chain` fails.

    Notes
    -----
    The unitarity of the chain for every :math:`N` follows from the algebraic
    certificate of the mode (thesis theorems; see the module docstring), which
    each mode re-checks numerically when its site tensor is built, where a
    check exists. The checks run here are independent *numerical* safeguards
    against implementation errors:

    * the normalization :math:`\operatorname{Tr}(U^\dagger U)/d^N = 1`
      (one sweep, any :math:`N`; necessary but not sufficient for
      unitarity);
    * when :math:`d^N \le` ``_DENSE_VERIFY_MAX_DIM``, the dense condition
      :math:`\|U^\dagger U - I\|_F \le 10^{-8}\sqrt{d^N}`.
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
    """Check normalization and, for small ``d**N``, dense unitarity of a chain.

    Parameters
    ----------
    tensors : list of ndarray
        Blocked site tensors ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors matching the outer bonds.

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
    # The dimension is computed from the shapes so that no large tensor is built.
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
    """Build a chain with a fixed or automatically chosen block size ``L``.

    ``one_blocking(sites_L, L)`` must return ``(tensors, l_vec, r_vec)`` for
    ``sites_L`` physical sites blocked by ``L``; ``sites_L`` is
    ``n_blocks * L`` or ``n_sites``. With ``blocking="auto"``, sizes
    ``L = 1, ..., max_blocking`` are tried in order and the first chain that
    satisfies Assumption 1 is returned; sizes ``L`` that do not divide
    ``n_sites`` are skipped. The size of the blocked tensor,
    ``(d**L)**2 * bond_dim**2`` complex numbers, is checked *before* it is
    built; the search stops at the first ``L`` above
    ``_AUTO_BLOCKING_MAX_BYTES``, and an explicit ``blocking=L`` above the cap
    raises.

    Parameters
    ----------
    one_blocking : callable
        ``one_blocking(sites_L, L) -> (tensors, l_vec, r_vec)``; builds,
        blocks and (optionally) verifies the chain of ``sites_L`` physical
        sites with block size ``L``. It raises ``ValueError`` for an
        inadmissible ``(sites_L, L)``.
    d : int
        Physical dimension.
    bond_dim : int
        Bond dimension (used for the size estimate).
    n_sites : int or None
        Number of physical sites; ``None`` if ``n_blocks`` is given.
    n_blocks : int or None
        Number of blocked tensors; ``None`` if ``n_sites`` is given.
    blocking : int or "auto"
        Block size, or ``"auto"`` for the search described above.
    max_blocking : int
        Largest ``L`` tried by ``"auto"``.

    Returns
    -------
    tuple
        ``(tensors, l_vec, r_vec)`` as returned by ``one_blocking``.

    Raises
    ------
    ValueError
        For ``"auto"``, if no admissible ``L`` gives a chain satisfying
        Assumption 1; for an explicit ``blocking``, if the blocked tensor
        exceeds the size cap (errors of ``one_blocking`` also propagate).
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


# --- MPUGenerator ---


class MPUGenerator:
    r"""Certified random generator of open-boundary MPUs (site-dependent).

    Every generation method returns ``(tensors, l_vec, r_vec)``: a list of
    site tensors ``A[p_in, p_out, bond_L, bond_R]`` and two boundary vectors
    whose contraction (module docstring) is unitary for every :math:`N`. The
    certificate behind this (proved for Modes I, II and grafted, validated
    numerically for Mode II-B) and the numerical checks run with
    ``verify=True`` are described in the module docstring.

    In this class every site draws its own Haar dressing;
    :class:`~mpu_generators.uniform.UniformMPUGenerator` draws one site tensor
    and reuses it at every site (translation invariance), and is otherwise
    identical.

    Parameters
    ----------
    seed : int
        Seed of the ``numpy.random.default_rng`` created at every call, so
        calling a method twice with the same arguments returns the same MPU.
    verify : bool
        Re-check the certificate of the mode numerically where a check exists
        (table in the module docstring) and run the safeguards of
        :func:`_finalize_chain` on the assembled chain.
    coherent_in_n : bool
        ``True`` (default): one random family for all chain lengths. The
        generator is seeded with ``seed`` alone, so for uniform generators the
        site tensor does not depend on ``n_sites``, and resampled chains of
        different lengths share their draws. ``False``: an independent draw
        for every chain length, seeded with ``(seed, n_sites)`` where
        ``n_sites`` counts physical sites before blocking.

    Attributes
    ----------
    seed : int
        As given.
    verify : bool
        As given.
    coherent_in_n : bool
        As given.

    Notes
    -----
    Sites are built in construction order and reversed at the end
    (:func:`_finalize_chain`). With per-site resampling, physical site
    :math:`j` of an :math:`N`-site chain therefore carries the
    :math:`(N-1-j)`-th draw: the chains for increasing :math:`N` extend a
    common sequence at the *left* end (Mode II-B: at the right end). For
    Mode I with ``readout``, the readout sites are drawn after the bulk
    sites, so they depend on :math:`N` in the resampled case.

    Examples
    --------
    >>> gen = MPUGenerator(seed=0)
    >>> tensors, l_vec, r_vec = gen.fusion(d=3, bond_dim=3, n_blocks=3)
    >>> len(tensors), tensors[0].shape
    (3, (3, 3, 3, 3))
    """

    #: ``False`` in :class:`UniformMPUGenerator`: one site tensor is drawn and
    #: reused at every site (translation invariance).
    _resample: bool = True

    def __init__(self, seed: int = 0, verify: bool = True, coherent_in_n: bool = True):
        self.seed = seed
        self.verify = verify
        self.coherent_in_n = coherent_in_n

    def _rng(self, n_sites_total: int) -> np.random.Generator:
        """Return the random generator for one chain.

        Parameters
        ----------
        n_sites_total : int
            Number of physical sites before blocking (only used if
            ``coherent_in_n`` is false).

        Returns
        -------
        numpy.random.Generator
            Seeded with ``seed``, or ``(seed, n_sites_total)`` if
            ``coherent_in_n`` is false.
        """
        if self.coherent_in_n:
            return np.random.default_rng(self.seed)
        return np.random.default_rng([self.seed, n_sites_total])

    def _sites(self, make_site, n_sites: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Return ``n_sites`` site tensors from ``make_site(rng)``.

        One call per site if ``_resample`` is true; otherwise one call whose
        result is repeated (the same array object at every position).

        Parameters
        ----------
        make_site : callable
            ``make_site(rng) -> ndarray``, one site tensor
            ``A[p_in, p_out, bond_L, bond_R]``.
        n_sites : int
            Number of sites.
        rng : numpy.random.Generator
            Passed to ``make_site``.

        Returns
        -------
        list of ndarray
            ``n_sites`` site tensors in construction order.
        """
        if self._resample:
            return [make_site(rng) for _ in range(n_sites)]
        site = make_site(rng)
        return [site] * n_sites

    # --- Blocking / auto-blocking driver (mode-agnostic) ---

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
        (full-rank boundary operators at every cut, required by
        ``TreeMPU.GeneralMPU(canonicalize=False)``) is returned. The projected
        size of each blocked tensor is checked *before* it is built
        (``_AUTO_BLOCKING_MAX_BYTES``).

        Parameters
        ----------
        d, bond_dim : int
            Physical and bond dimension.
        n_sites, n_blocks : int or None
            Give exactly one: number of physical sites, or number of blocked
            tensors.
        blocking : int or "auto"
            Block size, or ``"auto"`` for the search.
        max_blocking : int
            Largest block size tried by ``"auto"``.
        build_sites : callable
            ``build_sites(n_sites_total, rng) -> (sites, l_vec, r_vec)``:
            ``n_sites_total`` is the number of physical sites handed to the
            blocking step, ``rng`` comes from :meth:`_rng`, ``sites`` are in
            construction order (:func:`_finalize_chain` reverses them) and
            ``l_vec, r_vec`` are the boundary vectors.

        Returns
        -------
        tensors : list of ndarray
            Blocked site tensors.
        l_vec, r_vec : ndarray
            Boundary vectors.

        Raises
        ------
        ValueError
            If not exactly one of ``n_sites`` and ``n_blocks`` is given, if
            ``d < 2``, ``bond_dim < 1`` or ``n_blocks < 2``, if the chain has
            fewer than two blocked tensors or ``blocking`` does not divide the
            number of sites, or if the blocking search fails (see
            :func:`_run_blocking`).
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
            """Validate ``(sites_L, L)``, then build and finalize one chain."""
            if sites_L < 2:
                raise ValueError(f"n_sites must be >= 2, got {sites_L}.")
            if sites_L % L:
                raise ValueError(f"n_sites={sites_L} is not divisible by blocking={L}.")
            if sites_L // L < 2:
                raise ValueError(
                    f"n_sites={sites_L} with blocking={L} leaves "
                    f"{sites_L // L} tensor(s); need at least 2."
                )
            rng = self._rng(sites_L)
            sites, l_vec, r_vec = build_sites(sites_L, rng)
            return _finalize_chain(sites, l_vec, r_vec, L, self.verify)

        return _run_blocking(_one_blocking, d, bond_dim, n_sites, n_blocks, blocking, max_blocking)

    # --- Mode I: branching ---

    def branching(
        self,
        d: int = 2,
        bond_dim: int = 2,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
        readout: list[int] | None = None,
        digraph: dict[int, list[int]] | None = None,
        out_degree: int | str | None = None,
        ranks: str | dict[int, list[int]] = "balanced",
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Sample a Mode I (row resolution) MPU on a Haar-resolved digraph.

        The virtual bond indexes the ``bond_dim`` vertices of a digraph (by
        default random and out-regular with out-degree ``min(d, bond_dim)``
        and a balanced rank partition); every row of every site satisfies
        (RR) (:func:`_row_resolution_blocks`). With single-vertex readout,
        :math:`\ell=e_0` and :math:`r=(1,\dots,1)`, the chain is unitary for
        every :math:`N` (thesis ``thm:mode1-unitarity``); combined readout is
        covered by ``cor:mode1-readout``.

        Parameters
        ----------
        d : int
            Local physical dimension, at least 2.
        bond_dim : int
            Bond dimension :math:`\chi`, the number of digraph vertices.
        n_sites, n_blocks : int, optional
            Give exactly one: total number of physical sites, or number of
            tensors after blocking (then ``n_blocks * blocking`` sites). At
            least two tensors are required.
        blocking : int or "auto"
            Physical sites merged into each returned tensor. ``"auto"`` picks
            the smallest size satisfying Assumption 1 (see :meth:`_generate`).
            With ``readout``, ``blocking`` must divide ``m`` (see ``readout``).
        max_blocking : int
            Largest block size tried by ``blocking="auto"``.
        readout : list of int, optional
            Combined readout over the listed vertices (thesis
            ``cor:mode1-readout``, ``rem:cascaded-readout``): appends
            :math:`m=\max(1,\lceil\log_d |\mathrm{readout}|\rceil)` sites, a
            :math:`d`-ary tree of (RR) rows that resolves every listed vertex
            (:func:`_combined_readout_sites`), with :math:`\ell=(1)`. They come
            in addition to ``n_sites`` and are blocked with the rest, so
            ``blocking`` must divide ``n_sites`` and ``n_sites + m``, hence
            ``m`` (``m = 1``: only ``blocking = 1``; ``"auto"`` only tries
            divisors of ``m``). Default: single-vertex readout
            :math:`\ell=e_0`.
        digraph : dict, optional
            Hand-picked digraph ``{v: [targets]}`` on the vertices
            ``0..bond_dim-1``, used at every site; each vertex needs between 1
            and ``d`` distinct targets. Default: random (see ``out_degree``),
            redrawn at every site in the resampled (site-dependent) case and
            drawn once in the uniform case.
        out_degree : int or "random", optional
            Out-degree of the random digraph: a fixed value capped at
            ``min(d, bond_dim)`` (default: the cap), or ``"random"`` for a
            degree drawn uniformly in ``1..min(d, bond_dim)`` for every
            vertex. Ignored when ``digraph`` is given.
        ranks : "balanced", "random" or dict
            Rank partition of every row: as equal as possible (default), a
            uniformly random composition of ``d`` into positive parts (drawn
            per row and per site), or hand-picked ``{v: [ranks]}`` with one
            positive rank per target of ``v``, in the order of the digraph's
            target list, summing to ``d`` (requires ``digraph``).

        Returns
        -------
        tensors : list of ndarray
            Site tensors ``A[p_in, p_out, bond_L, bond_R]`` of shape
            ``(d**blocking, d**blocking, bond_dim, bond_dim)``; with
            ``readout`` the first tensor has ``bond_L = 1``.
        l_vec : ndarray
            Left boundary vector, shape ``(bond_dim,)`` (``(1,)`` with
            ``readout``).
        r_vec : ndarray
            Right boundary vector, shape ``(bond_dim,)``, all ones.

        Raises
        ------
        ValueError
            For invalid sizes (see :meth:`_generate`), an invalid ``digraph``
            or ``ranks``, ``ranks`` given as a dict without ``digraph``,
            ``out_degree < 1``, or ``blocking`` not dividing the chain length
            (with ``readout``: ``n_sites`` and ``m``).
        RuntimeError
            If ``verify`` is true and a numerical check fails.

        Notes
        -----
        Certified algebraically: every row of every drawn tensor (and of the
        readout cascade) satisfies (RR) for any Haar draw and any rank
        partition. The numerical re-check with ``verify=True``
        (:func:`_check_row_resolution`, :func:`_check_combined_readout_cascade`)
        is described in the module docstring. The non-default choices of
        ``digraph``, ``out_degree`` and ``ranks`` define different ensembles
        on the same certified class, so statistics are not comparable with the
        default one.
        """
        if digraph is not None:
            digraph = _check_digraph(digraph, bond_dim, d)
        if isinstance(ranks, dict):
            if digraph is None:
                raise ValueError("explicit ranks need an explicit digraph.")
            _check_ranks(ranks, digraph, d)

        def build_sites(n_sites_total: int, rng: np.random.Generator):
            # Callback of _generate: builds the bulk sites (and the readout
            # cascade) in construction order together with the boundary vectors.
            if digraph is None:
                edges = _random_out_regular_digraph(bond_dim, d, rng, out_degree)

            def make_site(r: np.random.Generator) -> np.ndarray:
                # One bulk tensor; a fresh random digraph per site only in the
                # resampled case, so uniform generators reuse ``edges``.
                if digraph is not None:
                    site_edges = digraph
                elif self._resample:
                    site_edges = _random_out_regular_digraph(bond_dim, d, r, out_degree)
                else:
                    site_edges = edges
                return _mode1_site_tensor(bond_dim, site_edges, d, r, ranks=ranks)

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
                # Reversed because _finalize_chain reverses the list, which puts
                # the cascade (level 0 first, bond_L = 1) at the left end.
                sites = sites + list(reversed(cascade_sites))
                l_vec = np.array([1.0], dtype=complex)

            r_vec = np.ones(bond_dim, dtype=complex)
            return sites, l_vec, r_vec

        return self._generate(d, bond_dim, n_sites, n_blocks, blocking, max_blocking, build_sites)

    # --- Mode II: fusion closure ---

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
        r"""Sample a Mode II (fusion closure) MPU from a bond-diagonal skeleton.

        The tensor is :math:`A[a,b]=\delta_{ab}\,y_a`, the channels close
        monomially, :math:`y_u^\dagger y_v\in\{0\}\cup\{y_w\}` (FC), the
        boundary :math:`\ell` lies on the skeleton's admissible manifold (B),
        and :math:`r=(1,\dots,1)`. Then
        :math:`U_N=\sum_u \ell_u\,y_u^{\otimes N}` is unitary for every
        :math:`N` (thesis ``thm:mode2-unitarity``). Each site conjugates all
        channels by one Haar unitary (the dressing), drawn per site, or once
        in the uniform case. The skeletons and their dimension constraints are
        tabulated in Notes; the checks run with ``verify=True`` are in the
        module docstring.

        Parameters
        ----------
        d, bond_dim, n_sites, n_blocks, blocking, max_blocking
            As in :meth:`branching`; ``bond_dim`` is the number of channels.
        skeleton : {"cyclic", "nested_flag", "matrix_units", "group"}
            Channel family (see Notes). Default ``"cyclic"``.
        group : {"Zn", "Zn_product", "S3"}
            Group for ``skeleton="group"`` (see :func:`_finite_group_table`);
            ignored otherwise.
        group_n : int or list of int
            Order of ``Zn``, or the factor orders of ``Zn_product``; ignored
            for ``S3`` and for the other skeletons. Default 3.

        Returns
        -------
        tensors : list of ndarray
            Site tensors ``A[p_in, p_out, bond_L, bond_R]`` of shape
            ``(d**blocking, d**blocking, bond_dim, bond_dim)``, bond-diagonal
            before blocking.
        l_vec : ndarray
            Left boundary vector, shape ``(bond_dim,)``.
        r_vec : ndarray
            Right boundary vector, shape ``(bond_dim,)``, all ones.

        Raises
        ------
        ValueError
            For invalid sizes (see :meth:`_generate`), an unknown ``skeleton``
            or ``group``, or dimensions incompatible with the skeleton (listed
            in Notes).
        RuntimeError
            If ``verify`` is true and a numerical check fails.

        Warns
        -----
        UserWarning
            If ``skeleton="cyclic"``, ``bond_dim > d`` and the chain has fewer
            than ``ceil((bond_dim - 1) / (d - 1))`` sites on each side of its
            middle cut, so that the bond dimension is not reached (see Notes).

        Notes
        -----
        Skeletons (builders in parentheses):

        ================  ================================================  ======================
        ``skeleton``      Channels; boundary (B)                            Constraint
        ================  ================================================  ======================
        ``cyclic``        :math:`\mathbb Z_M` characters                    none (see below)
                          (:func:`_cyclic_character_channels`);
                          flat Fourier spectrum
                          (:func:`_flat_spectrum_boundary`)
        ``nested_flag``   nested projectors, closure                        ``bond_dim <= d``
                          :math:`\max(u,v)`
                          (:func:`_nested_flag_channels`);
                          flat torus (:func:`_nested_flag_boundary`)
        ``matrix_units``  identity and :math:`m^2` matrix units             ``bond_dim = 1+m**2``,
                          :math:`y_{(i,j)}=w_iw_j^\dagger`                  ``m <= d``
                          (:func:`_matrix_unit_channels`);
                          Haar :math:`U(m)`
                          (:func:`_matrix_unit_boundary`)
        ``group``         left-regular representation of a                  ``d = bond_dim = |G|``
                          finite group (:func:`_group_regular_channels`);
                          Peter-Weyl manifold
                          :math:`\prod_\rho U(d_\rho)`
                          (:func:`_group_fourier_boundary`)
        ================  ================================================  ======================

        ``cyclic`` is a group-convolution skeleton only when ``bond_dim``
        divides ``d``. For ``bond_dim`` :math:`M>d` the channels are
        :math:`d\times d` matrices diagonal in one basis, hence linearly
        dependent, but the chain is still exactly unitary (the autocorrelation
        of a flat-spectrum ``l_vec`` is :math:`\delta_{w,0}`) and equals
        :math:`R\,e^{i\varphi(Q \bmod M)}R^\dagger` with :math:`Q` the total
        charge. Its operator-Schmidt rank across a cut is :math:`M` (for a
        generic boundary) once each side holds at least
        :math:`\lceil (M-1)/(d-1)\rceil` sites, because then the blocked
        channels :math:`y_g^{\otimes L}` are independent (Vandermonde on the
        values of :math:`Q`); shorter blocks do not satisfy Assumption 1, so
        use ``blocking="auto"`` or an explicit block of that size. This is
        the only check skipped for ``bond_dim > d``. ``nested_flag`` contains
        the MCZ / diagonal-phase family. ``matrix_units`` needs ``bond_dim`` = 2, 5, 10, 17, ...
        ``group="Zn_product"`` covers every finite abelian group and
        ``group="S3"`` the smallest non-abelian one.

        Certified algebraically: (FC) and (B) hold for the construction of each
        skeleton. The numerical re-check with ``verify=True`` is listed per
        skeleton in the module docstring.
        """
        if skeleton == "cyclic":
            # Any bond_dim; for bond_dim > d the channels are dependent by design
            # (see Notes) and a short chain cannot reach the requested bond dimension.
            n_phys = n_sites
            if n_phys is None and n_blocks is not None and isinstance(blocking, int):
                n_phys = n_blocks * blocking
            if bond_dim > d >= 2 and n_phys is not None:
                needed = -(-(bond_dim - 1) // (d - 1))  # ceil((bond_dim - 1) / (d - 1))
                if n_phys // 2 < needed:
                    reached = min(bond_dim, (n_phys // 2) * (d - 1) + 1)
                    warnings.warn(
                        f"skeleton='cyclic' with bond_dim={bond_dim} > d={d} needs at least "
                        f"{needed} sites on each side of a cut; with {n_phys} sites the "
                        f"operator-Schmidt rank is at most {reached}, so canonical form "
                        f"reduces the bond dimension.",
                        UserWarning,
                        stacklevel=2,
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
            # Callback of _generate: the boundary vector is drawn once per chain,
            # the dressed channel tensors by ``make_site`` (one per site, or one
            # repeated in the uniform case).
            if skeleton == "cyclic":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y = _cyclic_character_channels(d, bond_dim, r)
                    if self.verify:
                        _check_fusion_closure(y)
                        if bond_dim <= d:
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

                l_vec = _nested_flag_boundary(bond_dim, rng)

            elif skeleton == "matrix_units":

                def make_site(r: np.random.Generator) -> np.ndarray:
                    y, labels = _matrix_unit_channels(d, m, r)
                    if self.verify:
                        _check_matrix_unit_closure(y, labels)
                        if m < d:
                            # For m == d the identity equals sum_i y_(i,i): the
                            # channels are linearly dependent by design.
                            _check_channel_span_full_rank(y, bond_dim)
                    A = np.zeros((d, d, bond_dim, bond_dim), dtype=complex)
                    for a in range(bond_dim):
                        A[:, :, a, a] = y[a].T
                    return A

                l_vec = _matrix_unit_boundary(m, rng)

            else:  # group
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

    # --- Mode II-B: bond-resolved fusion (skeletons in skeletons.py) ---

    def fusion_bond_resolved(
        self,
        skeleton,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Sample a Mode II-B (bond-resolved fusion) MPU from a fixed skeleton.

        The core tensors and boundary vectors of ``skeleton`` are fixed; only
        the physical dressing :math:`B_u\mapsto(W_n\otimes\bar W_n)B_u` is
        drawn per site (thesis, Mode II-B sampling remark). The sites are
        taken from ``skeleton.dress(rng)``, one call per site (one call in the
        uniform case).

        Parameters
        ----------
        skeleton : mpu_generators.skeletons.FusionSkeleton
            Fixed core tensors and boundaries, e.g.
            :func:`mpu_generators.skeletons.lee_yang_skeleton`; its ``d``,
            ``bond_dim``, ``l_vec`` and ``r_vec`` define the chain.
        n_sites, n_blocks, blocking, max_blocking
            As in :meth:`branching`.

        Returns
        -------
        tensors : list of ndarray
            Site tensors ``A[p_in, p_out, bond_L, bond_R]`` of shape
            ``(skeleton.d**blocking, skeleton.d**blocking, skeleton.bond_dim,
            skeleton.bond_dim)``.
        l_vec, r_vec : ndarray
            Copies of ``skeleton.l_vec`` and ``skeleton.r_vec``.

        Raises
        ------
        ValueError
            If not exactly one of ``n_sites`` and ``n_blocks`` is given, for
            invalid sizes (as in :meth:`_generate`), or if the blocking search
            fails.
        RuntimeError
            If ``verify`` is true and the numerical safeguards fail.

        Notes
        -----
        The unitarity of Mode II-B chains is validated numerically only: no
        algebraic certificate is checked at generation time. With
        ``verify=True`` the chain passes the normalization check for any
        :math:`N` and the dense check
        :math:`\|U^\dagger U-1\|_F\le 10^{-8}\sqrt{d^N}` for
        :math:`d^N\le 2^{10}` (:func:`_verify_chain`); the dense check is the
        only direct test of unitarity. The intertwiner helpers
        :func:`mpu_generators.skeletons.solve_intertwiner` and
        :func:`mpu_generators.skeletons.verify_intertwiner` are standalone: the
        generator does not call them, and they can be run on a skeleton
        separately. The null-complement and boundary-compatibility conditions
        of the thesis are not checked by any helper.

        Unlike the other modes, the sites are used in draw order, without the
        reversal of :func:`_finalize_chain`, so for per-site resampling the
        chains for increasing :math:`N` extend a common sequence at the
        *right* end.
        """
        if (n_sites is None) == (n_blocks is None):
            raise ValueError("Give exactly one of n_sites or n_blocks.")

        def _one_blocking(sites_L: int, L: int):
            """Validate ``(sites_L, L)``, then dress, block and verify one chain."""
            if sites_L < 2:
                raise ValueError(f"n_sites must be >= 2, got {sites_L}.")
            if sites_L % L:
                raise ValueError(f"n_sites={sites_L} is not divisible by blocking={L}.")
            if sites_L // L < 2:
                raise ValueError(
                    f"n_sites={sites_L} with blocking={L} leaves "
                    f"{sites_L // L} tensor(s); need at least 2."
                )
            rng = self._rng(sites_L)
            sites = self._sites(lambda r: skeleton.dress(r), sites_L, rng)
            tensors = block_mpu_tensors(sites, L)
            l_vec, r_vec = skeleton.l_vec.copy(), skeleton.r_vec.copy()
            if self.verify:
                _verify_chain(tensors, l_vec, r_vec)
            return tensors, l_vec, r_vec

        return _run_blocking(
            _one_blocking, skeleton.d, skeleton.bond_dim, n_sites, n_blocks, blocking, max_blocking
        )

    # --- Grafted composition (Mode I inside Mode II) ---

    def grafted(
        self,
        d: int = 2,
        bond_dim: int = 2,
        n_sites: int | None = None,
        n_blocks: int | None = None,
        blocking: int | str = 1,
        max_blocking: int = 8,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        r"""Sample a grafted MPU (Mode I path inside Mode II channels).

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
            Bond dimension ``2 * n_channels`` (default 2, i.e. one channel);
            must be even with ``n_channels <= d``.

        Returns
        -------
        tensors : list of ndarray
            Site tensors ``A[p_in, p_out, bond_L, bond_R]`` of shape
            ``(d**blocking, d**blocking, bond_dim, bond_dim)``.
        l_vec : ndarray
            Left boundary vector, shape ``(bond_dim,)``; vanishes on the
            first ``n_channels`` entries.
        r_vec : ndarray
            Right boundary vector, shape ``(bond_dim,)``, all ones.

        Raises
        ------
        ValueError
            If ``bond_dim`` is odd, ``bond_dim // 2 > d``, or the sizes are
            invalid (see :meth:`_generate`).
        RuntimeError
            If ``verify`` is true and a numerical check fails.

        Notes
        -----
        Certified algebraically through its two factors: (RR) for the Mode I
        part and (FC) with a flat-spectrum boundary for the channels, each
        re-checked numerically at absolute tolerance ``1e-9`` when
        ``verify=True``. The channels are the cyclic skeleton
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
            # Callback of _generate: one flat-spectrum boundary per chain, one
            # grafted tensor per site (or one repeated, in the uniform case).
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
                            # K stores the transposed block and y[g] is an
                            # operator, so y[g].T @ K is the stored transpose
                            # of the operator K[a,ap] @ y[g].
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
    """Dispatch to :class:`MPUGenerator` or ``UniformMPUGenerator`` by ``mode``.

    Parameters
    ----------
    n_sites : int, optional
        Number of physical sites (give exactly one of ``n_sites`` and
        ``n_blocks``).
    d : int
        Local physical dimension.
    bond_dim : int
        Bond dimension.
    mode : {"branching", "mode1", "fusion", "mode2", "grafted"}
        Construction: :meth:`MPUGenerator.branching`,
        :meth:`MPUGenerator.fusion` or :meth:`MPUGenerator.grafted`. Mode II-B
        needs a ``FusionSkeleton`` object and is only available through
        :meth:`MPUGenerator.fusion_bond_resolved`.
    blocking : int or "auto"
        Physical sites per returned tensor.
    uniform : bool
        Use :class:`~mpu_generators.uniform.UniformMPUGenerator`
        (translation invariant) instead of :class:`MPUGenerator`.
    seed : int
        Seed of the generator.
    verify : bool
        Re-check the algebraic certificate numerically and run the numerical
        safeguards (see :class:`MPUGenerator`).
    n_blocks : int, optional
        Number of tensors after blocking, instead of ``n_sites``.
    max_blocking : int
        Largest block size tried by ``blocking="auto"``.
    **mode_kwargs
        Forwarded to the method, e.g. ``skeleton=`` for fusion or
        ``readout=`` for branching.

    Returns
    -------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or the selected method rejects its arguments.

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
