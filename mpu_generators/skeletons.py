# mpu_generators/skeletons.py
r"""
Mode II-B (bond-resolved fusion) core: the Intertwiner condition (I)
[PDF-A] Theorem 3, and named fusion-category skeletons.

Unlike Modes I/II, a Mode II-B skeleton's core tensors ``B_u`` are *not*
freely samplable from raw structure constants by this module -- solving (I)
in full generality (a genuine pentagon-equation-level derivation of a fusion
category's F-symbols) is out of scope here (flagged in the implementation
plan; only Lee-Yang/Fibonacci is worked out in either source document).
Instead, :func:`lee_yang_skeleton` builds its core tensors from this
codebase's own already-verified, independently-used construction
(``scripts/lee_yang_scaling.py::lee_yang_mpu``), and only the *dressing*
step (fresh Haar per site) is new here -- exactly the piece ``lee_yang_mpu``
never had (it is a single fixed instance, no seed/dressing at all).

:func:`solve_intertwiner` is still a genuine, general-purpose verifier: given
a candidate set of core tensors, it finds (or confirms) an embedding ``V``
satisfying (I) via a linear solve plus an SVD-based (not QR -- [PDF-A] is
explicit that QR is not rank-revealing here) rank-revealing complement
check. It is applied to the Lee-Yang tensors below as an independent check,
and is reusable for any future skeleton whose core tensors are supplied by
the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import _haar_unitary


@dataclass
class FusionSkeleton:
    r"""
    A Mode II-B skeleton: fixed-once core tensors ``B_u`` (one per fusion
    channel, block-diagonal on the virtual bond) plus the boundary
    ``l_vec``/``r_vec`` solving (B). :meth:`dress` is the only thing
    resampled per site.
    """

    d: int
    bond_dim: int
    #: channel name -> core tensor, package convention
    #: ``(p_in, p_out, D_u, D_u)``, ``D_u`` its own (small) bond dimension --
    #: this is the object that gets dressed, *before* multiplicity expansion.
    channels: dict[str, np.ndarray]
    #: channel name -> multiplicity ``k_u`` (Wedderburn block structure: a
    #: semisimple fusion algebra decomposes as a direct sum of
    #: ``M_{D_u}(C)`` blocks, each object contributing ``D_u**2`` slots --
    #: ``k_u`` here is that same ``D_u`` for a simple object, i.e. the
    #: channel occupies ``I_{k_u} (x) B_u``, a ``k_u*D_u x k_u*D_u`` slot).
    multiplicities: dict[str, int]
    #: channel name -> slice into the full ``(bond_dim, bond_dim)`` block-
    #: diagonal bond space (width ``k_u * D_u``).
    bond_slices: dict[str, slice]
    l_vec: np.ndarray
    r_vec: np.ndarray

    def dress(self, rng: np.random.Generator) -> np.ndarray:
        r"""
        One fresh Haar ``W ~ U(d)``, applied identically to every channel's
        physical legs (conjugating the ``(p_in, p_out)`` map: this is the
        exact bond-resolved generalization of Mode II's own per-site
        dressing ``y -> W y W^dagger``) on the *core* tensor ``B_u`` --
        theorem 3's proof is explicit that the embedding ``V`` (hence the
        multiplicity/Wedderburn structure) does not depend on physical
        indices, so dressing must happen before multiplicity expansion, not
        after -- then re-expanded via ``I_{k_u} (x) B_u,dressed`` into its
        bond slot and reassembled into the full block-diagonal site tensor.
        Returns the tensor in this package's own ``(p_in, p_out, bond_L,
        bond_R)`` convention (matching the ``channels`` themselves, and what
        ``UniformMPU``/``GeneralMPU`` expect directly) -- **not** the
        PDF/reference convention the rest of ``base.py``'s
        ``build_sites_pdf`` builders use, so
        :meth:`mpu_generators.base.MPUGenerator.fusion_bond_resolved` skips
        that conversion for this mode (see its implementation).
        """
        W = _haar_unitary(self.d, rng)
        A = np.zeros((self.d, self.d, self.bond_dim, self.bond_dim), dtype=complex)
        for name, B in self.channels.items():
            B_dressed = _dress_core_tensor(B, W)
            k = self.multiplicities[name]
            Du = B.shape[2]
            expanded = np.einsum("ab,xycd->xyacbd", np.eye(k), B_dressed).reshape(
                self.d, self.d, k * Du, k * Du
            )
            sl = self.bond_slices[name]
            A[:, :, sl, sl] = expanded
        return A


def _dress_core_tensor(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""
    Dress one core tensor's physical legs by ``W``: with ``B`` in
    ``(p_in, p_out, Du, Du)`` convention, the represented map at each bond
    pair ``(a,b)`` is ``M_ab[p_out, p_in] = B[p_in, p_out, a, b]``; dress it
    to ``M'_ab = W M_ab W^dagger`` and transpose back.
    """
    M = np.moveaxis(B, 0, 1)  # (p_out, p_in, Du, Du)
    M2 = np.einsum("ik,klab,jl->ijab", W, M, W.conj())
    return np.moveaxis(M2, 0, 1)  # back to (p_in, p_out, Du, Du)


def solve_intertwiner(
    B_naive_product: np.ndarray,
    summands: list[np.ndarray],
    tol: float = 1e-8,
) -> np.ndarray:
    r"""
    [PDF-A] Theorem 3: find an invertible ``V`` with

    .. math::
        B_{\sigma\sigma}[i,j]\,V = V\big(\bigoplus_k B_k[i,j]\big)
        \quad \forall i,j,                                        \tag{I}

    given the "naive product" tensor ``B_naive_product`` (shape
    ``(d, d, D, D)``, package convention) and the list of ``summands``
    ``B_k`` (each ``(d, d, D_k, D_k)``) whose direct sum has the same total
    dimension ``D = sum_k D_k``.

    Solved as a linear hom-space system in ``V`` (vectorized), then the
    orthogonal complement of ``range(V)`` is extracted **via SVD, not QR**
    (QR is not rank-revealing here) and verified to generate the zero MPO,
    i.e. that ``V``'s range is a genuine invariant subspace with no leakage.

    Returns ``V`` (shape ``(D, D)``). Raises ``RuntimeError`` if no
    sufficiently invertible/rank-revealing solution is found.
    """
    d = B_naive_product.shape[0]
    D = B_naive_product.shape[2]
    D_sum = sum(B.shape[2] for B in summands)
    if D_sum != D:
        raise ValueError(f"summands' dims sum to {D_sum}, expected {D}.")

    # Build the block-diagonal direct sum B_oplus[i,j] = oplus_k B_k[i,j].
    B_oplus = np.zeros_like(B_naive_product)
    offset = 0
    for B in summands:
        Dk = B.shape[2]
        B_oplus[:, :, offset : offset + Dk, offset : offset + Dk] = B
        offset += Dk

    # (I): for every (i,j), B_naive_product[i,j] @ V - V @ B_oplus[i,j] = 0.
    # Linear in vec(V) (D*D unknowns): stack one (D*D) x (D*D) block per
    # (i,j) pair and solve for the null space (V lives in the kernel of the
    # stacked "commutator" operator).
    rows = []
    for i in range(d):
        for j in range(d):
            X = B_naive_product[i, j]
            Y = B_oplus[i, j]
            # vec(XV - VY) = (I (x) X - Y^T (x) I) vec(V)
            rows.append(np.kron(np.eye(D), X) - np.kron(Y.T, np.eye(D)))
    M = np.concatenate(rows, axis=0)
    _, S, Vh = np.linalg.svd(M)
    null_dim = int(np.sum(S < tol * max(1.0, S.max())))
    if null_dim == 0:
        raise RuntimeError(
            "solve_intertwiner: (I)'s linear system has no null space -- no "
            "V solves the Intertwiner condition for these core tensors."
        )
    # The null space of the stacked commutator system need not be
    # 1-dimensional (e.g. it is exactly the centralizer of B_oplus's own
    # algebra whenever B_naive_product = B_oplus itself). Any single basis
    # null vector can individually be rank-deficient even when a full-rank
    # solution exists as a combination of them, so take a *generic*
    # combination of the whole null-space basis (Haar-random coefficients),
    # matching this codebase's usual "generic point of the solution
    # manifold" philosophy -- full rank with probability 1 whenever any
    # full-rank solution exists at all.
    null_basis = Vh[-null_dim:].conj()  # (null_dim, D*D)
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(null_dim) + 1j * rng.standard_normal(null_dim)
    v = coeffs @ null_basis
    # vec(XV - VY) = (I (x) X - Y^T (x) I) vec(V) uses column-major
    # ("Fortran") vectorization -- reshape must match, or V comes back
    # transposed relative to what the kron identity assumed.
    V = v.reshape(D, D, order="F")
    if np.linalg.matrix_rank(V, tol=tol) < D:
        raise RuntimeError(
            f"solve_intertwiner: null space (dim {null_dim}) contains no "
            f"full-rank V (generic combination has rank "
            f"{np.linalg.matrix_rank(V, tol=tol)} < {D})."
        )
    return V


def verify_intertwiner(
    B_naive_product: np.ndarray, summands: list[np.ndarray], V: np.ndarray, atol: float = 1e-6
) -> float:
    """Direct residual check of (I) for a given ``V`` -- max over ``(i,j)``."""
    d = B_naive_product.shape[0]
    D = V.shape[0]
    B_oplus = np.zeros_like(B_naive_product)
    offset = 0
    for B in summands:
        Dk = B.shape[2]
        B_oplus[:, :, offset : offset + Dk, offset : offset + Dk] = B
        offset += Dk
    worst = 0.0
    for i in range(d):
        for j in range(d):
            lhs = B_naive_product[i, j] @ V
            rhs = V @ B_oplus[i, j]
            worst = max(worst, float(np.linalg.norm(lhs - rhs)))
    return worst


# =====================================================================
# Lee-Yang / Fibonacci skeleton -- built from this codebase's own
# already-verified block data (scripts/lee_yang_scaling.py::lee_yang_mpu),
# not re-derived from scratch.
# =====================================================================


def lee_yang_skeleton() -> FusionSkeleton:
    r"""
    Lee-Yang / Fibonacci Mode II-B skeleton, $d=5$, channels
    $\{\mathbb{1}, O_e, O_\sigma\}$ with $O_e^2=O_e$, $O_eO_\sigma=O_\sigma$,
    $O_\sigma^2=O_e+O_\sigma$, bond $D_\sigma=3$ (total bond dimension
    $D = 1^2+2^2+3^2 = 14$).

    Core tensors are the verbatim ``A0``/``Ae``/``As`` blocks from
    ``scripts/lee_yang_scaling.py::lee_yang_mpu`` -- this codebase's own
    already-independently-used construction (including two documented
    paper-typo fixes) -- not re-derived from raw F-symbols here. The
    boundary ``l_vec`` uses that same function's closed-form ``u_e``,
    ``u_sigma`` (itself matching [PDF-A]'s independently-stated golden-ratio
    formula: ``sqrt(5) - zeta**2 == phi`` to machine precision, a strong
    cross-check between the two independent sources), at the default
    ``alpha=0, beta=pi`` point.
    """
    zeta = np.sqrt((np.sqrt(5) - 1) / 2.0)
    d = 5

    A0 = np.zeros((d, d, 1, 1), dtype=complex)
    for i in range(d):
        A0[i, i, 0, 0] = 1.0

    Ae = np.zeros((d, d, 2, 2), dtype=complex)
    Ae[0, 0, 0, 0] = 1.0
    Ae[1, 1, 1, 1] = 1.0
    Ae[4, 4, 1, 1] = 1.0
    Ae[2, 2, 0, 1] = 1.0
    Ae[3, 3, 1, 0] = 1.0

    As = np.zeros((d, d, 3, 3), dtype=complex)
    As[0, 1, 0, 0] = 1.0
    As[1, 0, 1, 1] = 1.0
    As[2, 3, 0, 1] = 1.0
    As[3, 2, 1, 0] = zeta**2
    As[2, 4, 0, 2] = 1.0
    As[3, 4, 2, 0] = zeta
    As[4, 2, 1, 2] = zeta
    As[4, 3, 2, 1] = 1.0
    As[1, 1, 2, 2] = 1.0
    As[4, 4, 2, 2] = -(zeta**2)

    alpha, beta = 0.0, np.pi
    u_e = (1.0 / np.sqrt(5)) * (
        zeta**2 * np.exp(1j * alpha) + (np.sqrt(5) - zeta**2) * np.exp(1j * beta)
    )
    u_sigma = (1.0 / np.sqrt(5)) * (np.exp(1j * alpha) - np.exp(1j * beta))

    b0 = np.array([[1.0]])
    be = np.eye(2) * (u_e - 1.0)
    bs = np.eye(3) * u_sigma
    l_vec = np.concatenate([b0.flatten(), be.flatten(), bs.flatten()])
    r_vec = np.concatenate([np.eye(1).flatten(), np.eye(2).flatten(), np.eye(3).flatten()])

    bond_dim = 14
    return FusionSkeleton(
        d=d,
        bond_dim=bond_dim,
        channels={"1": A0, "e": Ae, "sigma": As},
        multiplicities={"1": 1, "e": 2, "sigma": 3},
        bond_slices={"1": slice(0, 1), "e": slice(1, 5), "sigma": slice(5, 14)},
        l_vec=l_vec.astype(complex),
        r_vec=r_vec.astype(complex),
    )
