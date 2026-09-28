# mpu_generators/skeletons.py
r"""Mode II-B (bond-resolved fusion) skeletons and the intertwiner condition.

In Mode II-B every fusion channel :math:`u` is itself a small MPO with local
bond matrices :math:`B_u[i,j]\in M_{\chi_u}(\mathbb C)`; a skeleton fixes
these core tensors once, and only the physical dressing :math:`W\,\cdot\,W^\dagger`
is resampled per site (thesis, Ch. "Generating MPUs", Mode II-B).

Contents
--------
:class:`FusionSkeleton`
    Core tensors + block layout + boundary vectors; :meth:`FusionSkeleton.dress`
    produces one dressed site tensor.
:func:`mpo_product_tensor`
    The MPO-composition tensor :math:`B_{uv}[i,j]=\sum_k B_u[i,k]\otimes B_v[k,j]`
    (thesis ``eq:mpo-product-tensor``).
:func:`solve_intertwiner`, :func:`verify_intertwiner`
    Solve / check the bond-resolved intertwiner condition (I)
    (thesis ``eq:intertwiner``) for a full-column-rank map :math:`V`.
:func:`lee_yang_skeleton`
    The Lee-Yang (Fibonacci) skeleton, :math:`d=5`, :math:`\chi=14`.

Scope
-----
The thesis certificate for a bond-resolved skeleton has three parts:
(i) the intertwiner (I), (ii) the null-complement condition on
:math:`(\operatorname{im} V)^\perp`, and (iii) boundary compatibility
:math:`\mu_{\sigma\sigma}=V(\mu_a\oplus\mu_\sigma)`,
:math:`\lambda_{\sigma\sigma}^\dagger V=(\lambda_a\oplus\lambda_\sigma)^\dagger`.
This module implements (i) only. The Lee-Yang skeleton is built from
explicit core tensors and boundary data (thesis, Lee-Yang example); its
global unitarity is checked numerically by the generator
(``verify=True``), not derived from (i)-(iii) here. Mode II-B is outside
the scope of the tensor-first sampling paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import _haar_unitary


@dataclass
class FusionSkeleton:
    r"""A Mode II-B skeleton: fixed core tensors plus boundary vectors.

    The full site tensor is block diagonal on the virtual bond. Channel
    :math:`u` with core tensor :math:`B_u` (bond dimension :math:`\chi_u`)
    occupies the diagonal block :math:`I_{k_u}\otimes B_u` of width
    :math:`k_u\chi_u`, where :math:`k_u` is the multiplicity that realizes the
    trace-to-open-boundary step :math:`\operatorname{Tr}(\cdot)\to\ell^\top(\cdot)r`
    with the identity boundary on the :math:`k_u`-dimensional factor.

    Attributes
    ----------
    d : int
        Local physical dimension.
    bond_dim : int
        Total bond dimension :math:`\sum_u k_u\chi_u`.
    channels : dict[str, ndarray]
        Channel name -> core tensor, shape ``(d, d, chi_u, chi_u)`` in the
        package convention ``(p_in, p_out, bond_L, bond_R)``.
    multiplicities : dict[str, int]
        Channel name -> multiplicity :math:`k_u`.
    bond_slices : dict[str, slice]
        Channel name -> slice of the full bond space (width ``k_u * chi_u``).
    l_vec, r_vec : ndarray
        Boundary vectors of length ``bond_dim``.
    """

    d: int
    bond_dim: int
    channels: dict[str, np.ndarray]
    multiplicities: dict[str, int]
    bond_slices: dict[str, slice]
    l_vec: np.ndarray
    r_vec: np.ndarray

    def dress(self, rng: np.random.Generator) -> np.ndarray:
        r"""Return one site tensor with a fresh Haar dressing.

        A single :math:`W\sim\mathrm{Haar}(U(d))` conjugates the physical map
        of every channel, :math:`B_u[\cdot,\cdot]\mapsto W B_u W^\dagger`
        (the bond-resolved analogue of Mode II's :math:`y\mapsto WyW^\dagger`).
        Dressing acts on the physical legs and the multiplicity expansion on
        the bond legs, so the two operations commute.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of the Haar sample (one draw of :func:`_haar_unitary`).

        Returns
        -------
        ndarray
            Site tensor of shape ``(d, d, bond_dim, bond_dim)``.
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
    r"""Conjugate the physical map of a core tensor by ``W``.

    With ``B[p_in, p_out, a, b]`` the operator at bond entry ``(a, b)`` is
    :math:`M_{ab}[p_{out},p_{in}]=B[p_{in},p_{out},a,b]`; it becomes
    :math:`W M_{ab} W^\dagger`.
    """
    M = np.moveaxis(B, 0, 1)  # (p_out, p_in, a, b)
    M2 = np.einsum("ik,klab,jl->ijab", W, M, W.conj())
    return np.moveaxis(M2, 0, 1)  # back to (p_in, p_out, a, b)


def mpo_product_tensor(B_u: np.ndarray, B_v: np.ndarray) -> np.ndarray:
    r"""MPO-composition tensor of two channels (thesis ``eq:mpo-product-tensor``).

    Returns the local tensor of the operator product :math:`Y[u]\,Y[v]`
    (:math:`v` acts first), :math:`B_{uv}[i,j]=\sum_k B_u[i,k]\otimes B_v[k,j]`
    with :math:`i` the output and :math:`j` the input index.

    Parameters
    ----------
    B_u, B_v : ndarray
        Core tensors in package convention ``(p_in, p_out, chi, chi)``.

    Returns
    -------
    ndarray
        Shape ``(d, d, chi_u * chi_v, chi_u * chi_v)``, bond index
        ``a_u * chi_v + a_v`` (Kronecker order :math:`u\otimes v`).
    """
    d = B_u.shape[0]
    cu, cv = B_u.shape[2], B_v.shape[2]
    # package convention: B[p_in, p_out]; the product maps j (in) -> k -> i (out)
    P = np.einsum("kiab,jkcd->jiacbd", B_u, B_v)
    return P.reshape(d, d, cu * cv, cu * cv)


def _direct_sum(summands: list[np.ndarray]) -> np.ndarray:
    """Block-diagonal direct sum of core tensors on the bond legs."""
    d = summands[0].shape[0]
    D = sum(B.shape[2] for B in summands)
    out = np.zeros((d, d, D, D), dtype=complex)
    offset = 0
    for B in summands:
        Dk = B.shape[2]
        out[:, :, offset : offset + Dk, offset : offset + Dk] = B
        offset += Dk
    return out


def solve_intertwiner(
    B_product: np.ndarray,
    summands: list[np.ndarray],
    tol: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Find a full-column-rank intertwiner :math:`V` (thesis ``eq:intertwiner``).

    Solves

    .. math::
        B_{\sigma\sigma}[i,j]\,V = V\Big(\bigoplus_k B_k[i,j]\Big)
        \qquad \forall\, i,j, \tag{I}

    for :math:`V:\mathbb C^{\sum_k\chi_k}\to\mathbb C^{\chi_{\rm prod}}`.
    The dimensions need not agree (for Lee-Yang,
    :math:`\chi_\sigma^2=9` vs :math:`\chi_a+\chi_\sigma=5`); :math:`V` must
    be injective.

    Parameters
    ----------
    B_product : ndarray
        Product tensor, shape ``(d, d, chi_prod, chi_prod)`` (e.g. from
        :func:`mpo_product_tensor`).
    summands : list of ndarray
        Target channels :math:`B_k`, each ``(d, d, chi_k, chi_k)``.
    tol : float
        Relative singular-value threshold defining the solution space, and
        the rank threshold for injectivity.
    rng : numpy.random.Generator, optional
        Used to pick a generic element of the solution space. Defaults to
        ``numpy.random.default_rng(0)`` (deterministic).

    Returns
    -------
    ndarray
        :math:`V` of shape ``(chi_prod, sum_k chi_k)``, full column rank.

    Raises
    ------
    ValueError
        If ``sum_k chi_k > chi_prod`` (no injective map can exist).
    RuntimeError
        If (I) has no nonzero solution, or no injective one.

    Notes
    -----
    (I) is linear in :math:`V`: with column-major vectorization,
    :math:`\operatorname{vec}(XV-VY)=(I\otimes X-Y^\top\otimes I)\operatorname{vec}(V)`.
    The solution space is the null space of the stacked system over all
    :math:`(i,j)`, obtained by SVD. A basis vector of that space may be rank
    deficient even when an injective solution exists, so a generic (random)
    combination of the basis is returned; it is injective with probability 1
    whenever any injective solution exists.
    """
    d = B_product.shape[0]
    D_prod = B_product.shape[2]
    D_sum = sum(B.shape[2] for B in summands)
    if D_sum > D_prod:
        raise ValueError(
            f"summands' bond dimensions sum to {D_sum} > {D_prod}: no injective V exists."
        )
    B_oplus = _direct_sum(summands)

    rows = []
    for i in range(d):
        for j in range(d):
            X = B_product[i, j]
            Y = B_oplus[i, j]
            rows.append(np.kron(np.eye(D_sum), X) - np.kron(Y.T, np.eye(D_prod)))
    M = np.concatenate(rows, axis=0)
    _, S, Vh = np.linalg.svd(M)
    # Singular values beyond len(S) (when M has fewer rows than columns) are 0.
    n_unknowns = D_prod * D_sum
    null_dim = int(np.sum(S < tol * max(1.0, S.max()))) + (n_unknowns - len(S))
    if null_dim == 0:
        raise RuntimeError(
            "solve_intertwiner: the linear system (I) has only the trivial solution V = 0."
        )
    null_basis = Vh[-null_dim:].conj()  # (null_dim, D_prod * D_sum)
    rng = np.random.default_rng(0) if rng is None else rng
    coeffs = rng.standard_normal(null_dim) + 1j * rng.standard_normal(null_dim)
    V = (coeffs @ null_basis).reshape(D_prod, D_sum, order="F")
    rank = np.linalg.matrix_rank(V, tol=tol)
    if rank < D_sum:
        raise RuntimeError(
            f"solve_intertwiner: the solution space (dim {null_dim}) contains no "
            f"injective V (generic element has rank {rank} < {D_sum})."
        )
    return V


def verify_intertwiner(
    B_product: np.ndarray, summands: list[np.ndarray], V: np.ndarray, atol: float = 1e-6
) -> float:
    r"""Residual of (I) for a given :math:`V`.

    Returns
    -------
    float
        :math:`\max_{i,j}\|B_{\rm prod}[i,j]V - V B_\oplus[i,j]\|_F`.
        (``atol`` is accepted for backward compatibility and unused.)
    """
    d = B_product.shape[0]
    B_oplus = _direct_sum(summands)
    worst = 0.0
    for i in range(d):
        for j in range(d):
            worst = max(worst, float(np.linalg.norm(B_product[i, j] @ V - V @ B_oplus[i, j])))
    return worst


# =====================================================================
# Lee-Yang / Fibonacci skeleton
# =====================================================================


def lee_yang_skeleton(alpha: float = 0.0, beta: float = np.pi) -> FusionSkeleton:
    r"""Lee-Yang (Fibonacci) Mode II-B skeleton (thesis ``sec:lee-yang-worked``).

    Three channels :math:`\{1, O_e, O_\sigma\}` with fusion rules
    :math:`O_e^2=O_e`, :math:`O_eO_\sigma=O_\sigma`,
    :math:`O_\sigma^2=O_e+O_\sigma`; physical dimension :math:`d=5`, internal
    bond dimensions :math:`\chi_0,\chi_e,\chi_\sigma=1,2,3`. The trace-MPO
    channels are converted to open boundary as
    :math:`\tilde A_k=I_{\chi_k}\otimes A_k`, :math:`l_k=\operatorname{vec}(b_k)`,
    :math:`r_k=\operatorname{vec}(I_{\chi_k})`, giving total bond dimension
    :math:`1+4+9=14` and

    .. math::
        U_N = I^{\otimes N} + (u_e-1)\,O_{e,N} + u_\sigma\,O_{\sigma,N},

    with the two-parameter unitary boundary family

    .. math::
        u_e = \tfrac{1}{\sqrt5}\big(\zeta^2 e^{i\alpha} + \varphi e^{i\beta}\big),\qquad
        u_\sigma = \tfrac{1}{\sqrt5}\big(e^{i\alpha} - e^{i\beta}\big),

    :math:`\zeta^2=\varphi^{-1}=(\sqrt5-1)/2`.

    Parameters
    ----------
    alpha, beta : float
        Boundary phases. The default ``(0, pi)`` is the point used for all
        numerical results in the thesis.

    Returns
    -------
    FusionSkeleton
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
