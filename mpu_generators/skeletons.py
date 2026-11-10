# mpu_generators/skeletons.py
r"""Skeletons: the data of every construction, and the Mode II-B intertwiner tools.

A *skeleton* holds everything that defines one construction except the
random dressing and the pipeline: how the chain-level data are drawn
(:meth:`Skeleton.prepare`: a digraph, a boundary vector), how one site tensor
is drawn (:meth:`Skeleton.site`) and how the chain is closed
(:meth:`Skeleton.close`: boundary vectors, a readout cascade). The generators
own the pipeline (:meth:`~mpu_generators.base.MPUGenerator.sample`: size
checks, per-site or uniform sampling, blocking, verification), so every mode
runs through the same code and a new construction is one more subclass.

====================================  ======  ==========================================
class                                 mode    data
====================================  ======  ==========================================
:class:`RowResolutionSkeleton`        I       digraph, rank partition, readout
:class:`CyclicSkeleton`               II      :math:`\mathbb Z_M` characters
:class:`NestedFlagSkeleton`           II      shrinking-flag projectors
:class:`MatrixUnitSkeleton`           II      identity channel + matrix units
:class:`GroupSkeleton`                II      regular representation of a group
:class:`BondResolvedSkeleton`         II-B    per-channel core tensors (fixed)
:class:`GraftedSkeleton`              I+II    Mode I path inside cyclic channels
:class:`SharedBasisSkeleton`          I       counter automaton, one basis per site
:class:`MultiStartSkeleton`           I       copies of one automaton, one per guess
====================================  ======  ==========================================

The generator methods (``branching``, ``fusion``, ``fusion_bond_resolved``,
``grafted``, ``shared_basis_chain``, ``cayley_chain``, ``multi_start``) build the skeleton and
call ``sample``; ``sample(skeleton, ...)`` can be called directly, also with a
user-defined subclass.

Mode II-B
---------
In Mode II-B every fusion channel :math:`u` is itself a small MPO with local
bond matrices :math:`B_u[i,j]\in M_{\chi_u}(\mathbb C)`; a skeleton fixes
these core tensors once, and only the physical dressing :math:`W\,\cdot\,W^\dagger`
is resampled per site (thesis, Ch. "Generating MPUs", Mode II-B).

Contents
--------
:class:`BondResolvedSkeleton` (alias ``FusionSkeleton``)
    Core tensors + block layout + boundary vectors;
    :meth:`BondResolvedSkeleton.dress` produces one dressed site tensor.
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
:func:`solve_intertwiner` and :func:`verify_intertwiner` are standalone
helpers for (i); nothing in the package calls them at generation time, and
(ii) and (iii) are not checked anywhere. :func:`lee_yang_skeleton` builds
explicit core tensors and boundary data (thesis, Lee-Yang example). No
algebraic certificate is therefore checked when a Mode II-B chain is
generated: its unitarity is validated numerically only, by the normalization
check for any :math:`N` and, for :math:`d^N\le 2^{10}`, the dense check
:math:`\|U^\dagger U-1\|_F\le 10^{-8}\sqrt{d^N}` (both with
``verify=True``). Mode II-B is outside the scope of the tensor-first sampling
paper.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from .base import (
    _as_unitary,
    _check_channel_span_full_rank,
    _check_combined_readout_cascade,
    _check_digraph,
    _check_flat_spectrum,
    _check_fusion_closure,
    _check_fusion_closure_general,
    _check_group_boundary,
    _check_group_representation,
    _check_matrix_unit_closure,
    _check_nested_flag_closure,
    _check_ranks,
    _check_row_resolution,
    _combined_readout_sites,
    _cyclic_character_channels,
    _depth,
    _finite_group_table,
    _flat_spectrum_boundary,
    _group_fourier_boundary,
    _group_fusion_gamma,
    _group_regular_channels,
    _haar_unitary,
    _matrix_unit_boundary,
    _matrix_unit_channels,
    _mode1_site_tensor,
    _nested_flag_boundary,
    _nested_flag_channels,
    _random_out_regular_digraph,
    _random_transitions,
    _shared_basis_site,
)


class Skeleton(ABC):
    r"""Data of one construction; the generator owns the sampling pipeline.

    Subclasses define ``d`` and ``bond_dim`` (the largest bond dimension, used
    for size estimates) and three hooks. The generator calls them in this
    order for every chain: :meth:`prepare` once, :meth:`site` once per site
    (once in the uniform case), :meth:`close` once. Everything random must be
    drawn from the ``rng`` that is passed, in this order, so that the output
    depends only on the seed.

    Attributes
    ----------
    d : int
        Physical dimension.
    bond_dim : int
        Largest bond dimension.
    reverse_sites : bool
        ``True`` (all modes except II-B): the sites are drawn in construction
        order and reversed at the end, so that the chains for increasing
        :math:`N` extend a common sequence at the left end. ``False``
        (Mode II-B, kept for reproducibility of the thesis instances): drawn
        order is physical order.
    """

    d: int
    bond_dim: int
    reverse_sites: bool = True

    @abstractmethod
    def prepare(self, rng: np.random.Generator, n_sites: int, uniform: bool, verify: bool) -> Any:
        """Draw the chain-level data (digraph, boundary) and return the state.

        ``n_sites`` is the number of physical sites handed to the blocking
        step; ``uniform`` is true when one site is reused at every position.
        """

    @abstractmethod
    def site(self, state: Any, rng: np.random.Generator, k: int, verify: bool) -> np.ndarray:
        """Draw one site tensor ``A[p_in, p_out, bond_L, bond_R]``.

        ``k`` is the physical position of the site (``0`` is the left end);
        it only matters for site-dependent bond dimensions.
        """

    @abstractmethod
    def close(
        self, state: Any, sites: list[np.ndarray], rng: np.random.Generator, verify: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Return ``(sites, l_vec, r_vec)`` in construction order.

        May append sites (the readout cascade of Mode I) and draw from ``rng``
        after the bulk sites.
        """


@dataclass
class BondResolvedSkeleton(Skeleton):
    r"""A Mode II-B skeleton: fixed core tensors plus boundary vectors.

    The full site tensor is block diagonal on the virtual bond. Channel
    :math:`u` with core tensor :math:`B_u` (bond dimension :math:`\chi_u`)
    occupies the diagonal block :math:`I_{k_u}\otimes B_u` of width
    :math:`k_u\chi_u`, where :math:`k_u` is the multiplicity that realizes the
    trace-to-open-boundary step :math:`\operatorname{Tr}(\cdot)\to\ell^\top(\cdot)r`
    with the identity boundary on the :math:`k_u`-dimensional factor.

    Parameters
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

    Attributes
    ----------
    d, bond_dim, channels, multiplicities, bond_slices, l_vec, r_vec
        Stored unchanged under the names of the constructor parameters.
    """

    d: int
    bond_dim: int
    channels: dict[str, np.ndarray]
    multiplicities: dict[str, int]
    bond_slices: dict[str, slice]
    l_vec: np.ndarray
    r_vec: np.ndarray

    #: sites are used in draw order (see :class:`Skeleton`)
    reverse_sites = False

    def prepare(self, rng, n_sites, uniform, verify):
        """Nothing is drawn at chain level: the cores and boundaries are fixed."""

    def site(self, state, rng, k, verify):
        """One dressed site tensor (:meth:`dress`)."""
        return self.dress(rng)

    def close(self, state, sites, rng, verify):
        """Copies of the fixed boundary vectors."""
        return sites, self.l_vec.copy(), self.r_vec.copy()

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

    Parameters
    ----------
    B : ndarray
        Core tensor, shape ``(d, d, chi, chi)`` in the package convention.
    W : ndarray
        Unitary of shape ``(d, d)``.

    Returns
    -------
    ndarray
        Dressed core tensor, same shape as ``B``.
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
    """Return the block-diagonal direct sum of core tensors on the bond legs.

    Parameters
    ----------
    summands : list of ndarray
        Core tensors, each ``(d, d, chi_k, chi_k)`` with a common ``d``.

    Returns
    -------
    ndarray
        Shape ``(d, d, sum_k chi_k, sum_k chi_k)``.
    """
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
        Threshold on the singular values of the stacked system (relative to
        ``max(1, largest singular value)``) that defines the solution space;
        also the absolute rank threshold used to test that :math:`V` is
        injective.
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
    r"""Return the residual of the intertwiner condition (I) for a given :math:`V`.

    Parameters
    ----------
    B_product : ndarray
        Product tensor, shape ``(d, d, chi_prod, chi_prod)``.
    summands : list of ndarray
        Target channels :math:`B_k`, each ``(d, d, chi_k, chi_k)``.
    V : ndarray
        Candidate intertwiner, shape ``(chi_prod, sum_k chi_k)``.
    atol : float
        Unused; has no effect on the result.

    Returns
    -------
    float
        :math:`\max_{i,j}\|B_{\rm prod}[i,j]V - V B_\oplus[i,j]\|_F`.
        This is a diagnostic only: the caller compares it with a tolerance.
    """
    d = B_product.shape[0]
    B_oplus = _direct_sum(summands)
    worst = 0.0
    for i in range(d):
        for j in range(d):
            worst = max(worst, float(np.linalg.norm(B_product[i, j] @ V - V @ B_oplus[i, j])))
    return worst


# --- Lee-Yang / Fibonacci skeleton ---


def lee_yang_skeleton(alpha: float = 0.0, beta: float = np.pi) -> BondResolvedSkeleton:
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
    BondResolvedSkeleton
        Skeleton with ``d = 5``, ``bond_dim = 14``, channels ``"1"``, ``"e"``,
        ``"sigma"`` (multiplicities 1, 2, 3) and boundary vectors of length 14.

    Notes
    -----
    The core tensors and boundaries are explicit; this function does not call
    :func:`solve_intertwiner` or :func:`verify_intertwiner`. No algebraic
    certificate is checked for the generated chains; their unitarity is
    validated numerically by the generator (``verify=True``).
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
    return BondResolvedSkeleton(
        d=d,
        bond_dim=bond_dim,
        channels={"1": A0, "e": Ae, "sigma": As},
        multiplicities={"1": 1, "e": 2, "sigma": 3},
        bond_slices={"1": slice(0, 1), "e": slice(1, 5), "sigma": slice(5, 14)},
        l_vec=l_vec.astype(complex),
        r_vec=r_vec.astype(complex),
    )


#: Backward-compatible name of :class:`BondResolvedSkeleton` (Mode II-B).
FusionSkeleton = BondResolvedSkeleton


# --- Mode I: row resolution ---


class RowResolutionSkeleton(Skeleton):
    r"""Mode I: a digraph on the bond dimension whose rows resolve the identity.

    Parameters
    ----------
    d, bond_dim : int
        Local and bond dimension.
    digraph : dict, optional
        Hand-picked ``{v: [targets]}`` used at every site; random
        out-regular digraph otherwise (fresh per site for the site-dependent
        generator, one for the chain for the uniform one).
    out_degree : int or "random", optional
        Out-degree of the random digraph.
    ranks : "balanced", "random" or dict
        Rank partition of every row (needs ``digraph`` when a dict).
    readout : list of int, optional
        Vertices of a combined multi-vertex readout (appends a cascade of
        sites and makes the left bond one-dimensional).

    Raises
    ------
    ValueError
        For an invalid ``digraph`` or ``ranks``, or ``ranks`` as a dict
        without ``digraph``.
    """

    def __init__(self, d, bond_dim, digraph=None, out_degree=None, ranks="balanced", readout=None):
        self.d, self.bond_dim = d, bond_dim
        self.digraph = None if digraph is None else _check_digraph(digraph, bond_dim, d)
        if isinstance(ranks, dict):
            if self.digraph is None:
                raise ValueError("explicit ranks need an explicit digraph.")
            _check_ranks(ranks, self.digraph, d)
        self.out_degree, self.ranks, self.readout = out_degree, ranks, readout

    def prepare(self, rng, n_sites, uniform, verify):
        """Draw the chain digraph (also drawn, and unused, for per-site sampling)."""
        edges = None
        if self.digraph is None:
            edges = _random_out_regular_digraph(self.bond_dim, self.d, rng, self.out_degree)
        return SimpleNamespace(edges=edges, uniform=uniform)

    def site(self, state, rng, k, verify):
        """One Mode I tensor; (RR) is re-checked when ``verify``."""
        if self.digraph is not None:
            edges = self.digraph
        elif not state.uniform:
            edges = _random_out_regular_digraph(self.bond_dim, self.d, rng, self.out_degree)
        else:
            edges = state.edges
        A = _mode1_site_tensor(self.bond_dim, edges, self.d, rng, ranks=self.ranks)
        if verify:
            _check_row_resolution(A)
        return A

    def close(self, state, sites, rng, verify):
        """Boundary ``e_0`` and ones, or the readout cascade on the left."""
        if self.readout is None:
            l_vec = np.zeros(self.bond_dim, dtype=complex)
            l_vec[0] = 1.0
        else:
            cascade = _combined_readout_sites(self.readout, self.d, self.bond_dim, rng)
            if verify:
                _check_combined_readout_cascade(cascade)
            # _finalize_chain reverses the list: the cascade (level 0 first,
            # bond_L = 1) ends up at the left end.
            sites = sites + list(reversed(cascade))
            l_vec = np.array([1.0], dtype=complex)
        return sites, l_vec, np.ones(self.bond_dim, dtype=complex)


# --- Mode II: fusion closure (bond-diagonal channels) ---


class FusionClosureSkeleton(Skeleton):
    r"""Mode II base class: bond-diagonal tensor :math:`A[a,a]=y_a` with closed channels.

    Subclass it to add a fusion rule. The channels :math:`y_a` of a site and
    the boundary :math:`\ell` are the data; :math:`r=(1,\dots,1)`. A subclass
    implements :meth:`setup` (chain-level draws and tables, returning a
    namespace with at least ``l_vec``), :meth:`channels` (one site, returning
    ``(y, aux)`` with ``y[a]`` the operators) and, optionally,
    :meth:`check_channels`.
    """

    def __init__(self, d, bond_dim):
        self.d, self.bond_dim = d, bond_dim

    @abstractmethod
    def setup(self, rng, verify):
        """Draw the boundary (and build tables); return a namespace with ``l_vec``."""

    @abstractmethod
    def channels(self, state, rng):
        """Return ``(y, aux)``: the channel operators of one site and extra data."""

    def check_channels(self, state, y, aux):
        """Re-check the algebraic certificate of one site (default: none)."""

    def prepare(self, rng, n_sites, uniform, verify):
        return self.setup(rng, verify)

    def site(self, state, rng, k, verify):
        y, aux = self.channels(state, rng)
        if verify:
            self.check_channels(state, y, aux)
        A = np.zeros((self.d, self.d, self.bond_dim, self.bond_dim), dtype=complex)
        for a in range(self.bond_dim):
            A[:, :, a, a] = y[a].T
        return A

    def close(self, state, sites, rng, verify):
        return sites, state.l_vec, np.ones(self.bond_dim, dtype=complex)


class CyclicSkeleton(FusionClosureSkeleton):
    r""":math:`\mathbb Z_M` characters, any ``bond_dim`` (also ``bond_dim > d``)."""

    def setup(self, rng, verify):
        l_vec = _flat_spectrum_boundary(self.bond_dim, rng)
        if verify:
            _check_flat_spectrum(l_vec)
        return SimpleNamespace(l_vec=l_vec)

    def channels(self, state, rng):
        return _cyclic_character_channels(self.d, self.bond_dim, rng), None

    def check_channels(self, state, y, aux):
        _check_fusion_closure(y)
        if self.bond_dim <= self.d:
            _check_channel_span_full_rank(y, self.bond_dim)


class NestedFlagSkeleton(FusionClosureSkeleton):
    """Shrinking-flag projectors (contains MCZ); needs ``d >= bond_dim``."""

    def __init__(self, d, bond_dim):
        if bond_dim > d:
            raise ValueError(
                f"skeleton='nested_flag' needs d >= bond_dim (got d={d}, bond_dim={bond_dim})."
            )
        super().__init__(d, bond_dim)

    def setup(self, rng, verify):
        return SimpleNamespace(l_vec=_nested_flag_boundary(self.bond_dim, rng))

    def channels(self, state, rng):
        return _nested_flag_channels(self.d, self.bond_dim, rng), None

    def check_channels(self, state, y, aux):
        _check_nested_flag_closure(y)
        _check_channel_span_full_rank(y, self.bond_dim)


class MatrixUnitSkeleton(FusionClosureSkeleton):
    """Identity channel plus matrix units; ``bond_dim = 1 + m^2`` with ``m <= d``."""

    def __init__(self, d, bond_dim):
        m = math.isqrt(bond_dim - 1) if bond_dim >= 1 else -1
        if m < 1 or 1 + m * m != bond_dim:
            raise ValueError(
                f"skeleton='matrix_units' needs bond_dim = 1 + m^2 for "
                f"an integer m >= 1 (got bond_dim={bond_dim}); valid "
                f"values are 2, 5, 10, 17, 26, ..."
            )
        if m > d:
            raise ValueError(
                f"skeleton='matrix_units' needs m <= d (got m={m} from bond_dim={bond_dim}, d={d})."
            )
        super().__init__(d, bond_dim)
        self.m = m

    def setup(self, rng, verify):
        return SimpleNamespace(l_vec=_matrix_unit_boundary(self.m, rng))

    def channels(self, state, rng):
        return _matrix_unit_channels(self.d, self.m, rng)

    def check_channels(self, state, y, aux):
        _check_matrix_unit_closure(y, aux)
        if self.m < self.d:
            # For m == d the identity equals sum_i y_(i,i): the channels are
            # linearly dependent by design.
            _check_channel_span_full_rank(y, self.bond_dim)


class GroupSkeleton(FusionClosureSkeleton):
    """Regular representation of a finite group (``Zn``, ``Zn_product``, ``S3``)."""

    def __init__(self, d, bond_dim, group="Zn", group_n=3):
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
        super().__init__(d, bond_dim)
        self.group, self.group_n = group, group_n

    def setup(self, rng, verify):
        elements, mult, inv, identity, irreps = _finite_group_table(self.group, self.group_n)
        if verify:
            _check_group_representation(elements, mult, irreps)
        index = {g: i for i, g in enumerate(elements)}
        l_vec = _group_fourier_boundary(elements, irreps, rng)
        if verify:
            _check_group_boundary(l_vec, elements, mult, index, identity)
        return SimpleNamespace(
            l_vec=l_vec,
            y_group=_group_regular_channels(elements, mult),
            gamma=_group_fusion_gamma(elements, mult, inv),
        )

    def channels(self, state, rng):
        W = _haar_unitary(self.d, rng)
        return np.einsum("ij,ajk,lk->ail", W, state.y_group, W.conj()), None

    def check_channels(self, state, y, aux):
        _check_fusion_closure_general(y, state.gamma)
        _check_channel_span_full_rank(y, self.bond_dim)


# --- Grafted: a Mode I path inside cyclic channels ---


class GraftedSkeleton(Skeleton):
    r"""A two-vertex Mode I path grafted into the channels of the cyclic skeleton.

    ``bond_dim = 2 * n_channels`` (even, ``n_channels <= d``).
    """

    _K_EDGES: ClassVar[dict[int, list[int]]] = {0: [0], 1: [1, 0]}

    def __init__(self, d, bond_dim):
        if bond_dim % 2:
            raise ValueError(f"mode='grafted' has bond_dim = 2 * n_channels; got odd {bond_dim}.")
        n_channels = bond_dim // 2
        if n_channels > d:
            raise ValueError(
                f"mode='grafted' needs d >= bond_dim // 2 (got d={d}, "
                f"bond_dim={bond_dim} => n_channels={n_channels})."
            )
        self.d, self.bond_dim, self.n_channels = d, bond_dim, n_channels

    def prepare(self, rng, n_sites, uniform, verify):
        l_conv = _flat_spectrum_boundary(self.n_channels, rng)
        if verify:
            _check_flat_spectrum(l_conv)
        return l_conv

    def site(self, state, rng, k, verify):
        n_channels, d = self.n_channels, self.d
        K = _mode1_site_tensor(2, self._K_EDGES, d, rng, identity_rows=(0,))
        y = _cyclic_character_channels(d, n_channels, rng)
        if verify:
            _check_row_resolution(K)
            _check_fusion_closure(y)
            _check_channel_span_full_rank(y, n_channels)
        A = np.zeros((d, d, self.bond_dim, self.bond_dim), dtype=complex)
        for a in range(2):
            for ap in range(2):
                if not np.any(K[:, :, a, ap]):
                    continue
                for g in range(n_channels):
                    # K stores the transposed block and y[g] is an operator,
                    # so y[g].T @ K is the stored transpose of K[a,ap] @ y[g].
                    A[:, :, a * n_channels + g, ap * n_channels + g] = y[g].T @ K[:, :, a, ap]
        return A

    def close(self, state, sites, rng, verify):
        l_vec = np.zeros(self.bond_dim, dtype=complex)
        l_vec[self.n_channels :] = state
        return sites, l_vec, np.ones(self.bond_dim, dtype=complex)


# --- Shared-basis counter automata ---


class SharedBasisSkeleton(Skeleton):
    r"""Finite-state (counter / group) automaton with one reading basis per site.

    See :meth:`~mpu_generators.base.MPUGenerator.shared_basis_chain` for the
    construction, the unitarity argument and the parameters.
    """

    def __init__(
        self,
        d,
        D=4,
        transitions=None,
        R=None,
        V=None,
        final_phases=None,
        start=0,
    ):
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}.")
        self.d = d
        self.D = D if isinstance(D, (int, np.integer)) else [int(x) for x in D]
        dims = [self.D] if isinstance(self.D, int) else self.D
        if min(dims) < 1:
            raise ValueError("bond dimensions must be >= 1.")
        self.bond_dim = max(dims)
        if isinstance(transitions, str) and transitions not in ("permutation", "ring", "random"):
            raise ValueError(
                f"unknown transitions {transitions!r}; use 'permutation', 'ring', "
                "'random' or an explicit list."
            )
        if isinstance(final_phases, str) and final_phases != "random":
            raise ValueError("final_phases must be an array, None or 'random'.")
        self.transitions, self.R, self.V = transitions, R, V
        self.final_phases, self.start = final_phases, start

    def _per_site(self, value, depth_single, name, n, uniform):
        if value is None:
            return None
        depth = _depth(value)
        if depth == depth_single:
            return [value] * n
        if depth == depth_single + 1:
            if uniform:
                raise ValueError(
                    f"{name} per site needs the site-dependent MPUGenerator; "
                    "UniformMPUGenerator reuses one tensor."
                )
            if len(value) != n:
                raise ValueError(f"{name} lists {len(value)} sites; the chain has {n}.")
            return list(value)
        raise ValueError(f"{name} has an unexpected nesting depth.")

    def prepare(self, rng, n_sites, uniform, verify):
        n = n_sites
        if isinstance(self.D, int):
            Ds = [self.D] * (n + 1)
        else:
            if uniform:
                raise ValueError("a translation-invariant generator needs an integer D.")
            Ds = list(self.D)
            if len(Ds) != n + 1:
                raise ValueError(
                    f"D must list {n + 1} bond dimensions for {n} sites; got {len(Ds)}."
                )
        if not 0 <= self.start < Ds[0]:
            raise ValueError(f"start={self.start} out of range for D_0={Ds[0]}.")
        mode, tr_list = None, None
        if isinstance(self.transitions, str):
            mode = self.transitions
        elif self.transitions is None:
            mode = "permutation" if len(set(Ds)) == 1 else "random"
        else:
            tr_list = self._per_site(self.transitions, 2, "transitions", n, uniform)
        return SimpleNamespace(
            Ds=Ds,
            n=n,
            uniform=uniform,
            mode=mode,
            tr_list=tr_list,
            R_list=self._per_site(self.R, 2, "R", n, uniform),
            V_list=self._per_site(self.V, 3, "V", n, uniform),
            data=None,
        )

    def _automaton(self, state, rng, k):
        """Return the maps, the basis and the unitaries of site ``k`` (drawn once if uniform)."""
        d = self.d
        D_in, D_out = state.Ds[k], state.Ds[k + 1]
        if state.data is None or not state.uniform:
            if state.tr_list is not None:
                t = [[int(u) for u in row] for row in state.tr_list[k]]
                if len(t) != d or any(len(row) != D_in for row in t):
                    raise ValueError(f"transitions at site {k} must be {d} lists of length {D_in}.")
                if any(not 0 <= u < D_out for row in t for u in row):
                    raise ValueError(f"transitions entries at site {k} must lie in 0..{D_out - 1}.")
            else:
                t = _random_transitions(state.mode, d, D_in, D_out, rng)
            Rk = (
                _haar_unitary(d, rng)
                if state.R_list is None
                else _as_unitary(state.R_list[k], d, "R")
            )
            if state.V_list is None:
                Vk = [_haar_unitary(d, rng) for _ in range(D_in)]
            else:
                if len(state.V_list[k]) != D_in:
                    raise ValueError(
                        f"V at site {k} must contain D_{k}={D_in} unitaries; "
                        f"got {len(state.V_list[k])}."
                    )
                Vk = [_as_unitary(Vv, d, "V[v]") for Vv in state.V_list[k]]
            state.data = (t, Rk, Vk)
        return state.data

    def site(self, state, rng, k, verify):
        t, Rk, Vk = self._automaton(state, rng, k)
        A = _shared_basis_site(t, Rk, Vk, self.d, state.Ds[k + 1])
        if verify:
            _check_row_resolution(A)
        return A

    def close(self, state, sites, rng, verify):
        D_end = state.Ds[-1]
        if self.final_phases is None:
            phases = np.zeros(D_end)
        elif isinstance(self.final_phases, str):
            phases = rng.uniform(0, 2 * np.pi, size=D_end)
        else:
            phases = np.asarray(self.final_phases, dtype=float)
            if phases.shape != (D_end,):
                raise ValueError(f"final_phases must have shape ({D_end},); got {phases.shape}.")
        l_vec = np.zeros(state.Ds[0], dtype=complex)
        l_vec[self.start] = 1.0
        return sites, l_vec, np.exp(1j * phases)


class MultiStartSkeleton(SharedBasisSkeleton):
    r"""Mode I with several starts: copies of one automaton, one per guess of the charge.

    See :meth:`~mpu_generators.base.MPUGenerator.multi_start` for the
    construction, the unitarity argument and the parameters. The bond is block
    diagonal, one block per pair (guess, Fourier index); every block is a
    shared-basis chain.
    """

    def __init__(
        self,
        d,
        n_sectors=2,
        D=4,
        transitions=None,
        R=None,
        V=None,
        final_phases=None,
        start=0,
        charges=None,
        sector_phases="random",
        edge_phases="random",
    ):
        super().__init__(d, D, transitions, R, V, final_phases, start)
        if n_sectors < 1:
            raise ValueError(f"n_sectors must be >= 1; got {n_sectors}.")
        self.n_sectors = int(n_sectors)
        self.charges = np.arange(d) % n_sectors if charges is None else np.asarray(charges)
        if self.charges.shape != (d,) or not np.issubdtype(self.charges.dtype, np.integer):
            raise ValueError(f"charges must be {d} integers.")
        if isinstance(sector_phases, str) and sector_phases != "random":
            raise ValueError("sector_phases must be an array or 'random'.")
        if isinstance(edge_phases, str) and edge_phases != "random":
            raise ValueError("edge_phases must be an array, None or 'random'.")
        self.sector_phases, self.edge_phases = sector_phases, edge_phases
        # Without edge phases all guesses are the same tensor: one block per Fourier index.
        self.n_guesses = 1 if edge_phases is None else self.n_sectors
        self.n_copies = self.n_guesses * self.n_sectors
        self.bond_dim *= self.n_copies

    def prepare(self, rng, n_sites, uniform, verify):
        state = super().prepare(rng, n_sites, uniform, verify)
        m = self.n_sectors
        if isinstance(self.sector_phases, str):
            state.phi = rng.uniform(0, 2 * np.pi, size=m)
        else:
            state.phi = np.asarray(self.sector_phases, dtype=float)
            if state.phi.shape != (m,):
                raise ValueError(f"sector_phases must have shape ({m},); got {state.phi.shape}.")
        state.theta = None
        return state

    def _edge_phases(self, state, rng, D_in):
        """Return ``theta[guess, v, a]`` of one site (drawn once if uniform)."""
        shape = (self.n_guesses, D_in, self.d)
        if self.edge_phases is None:
            return np.zeros(shape)
        if isinstance(self.edge_phases, str):
            if state.theta is None or not state.uniform:
                state.theta = rng.uniform(0, 2 * np.pi, size=shape)
            return state.theta
        theta = np.asarray(self.edge_phases, dtype=float)
        if theta.shape != shape:
            raise ValueError(f"edge_phases must have shape {shape}; got {theta.shape}.")
        return theta

    def site(self, state, rng, k, verify):
        d, m = self.d, self.n_sectors
        D_in, D_out = state.Ds[k], state.Ds[k + 1]
        t, Rk, Vk = self._automaton(state, rng, k)
        theta = self._edge_phases(state, rng, D_in)
        A = np.zeros((d, d, self.n_copies * D_in, self.n_copies * D_out), dtype=complex)
        for b in range(self.n_copies):
            guess, g = divmod(b, m)
            sign = np.exp(2j * np.pi * g * self.charges / m)
            block = A[:, :, b * D_in : (b + 1) * D_in, b * D_out : (b + 1) * D_out]
            for v, Vv in enumerate(Vk):
                for a in range(d):
                    ra = Rk[:, a : a + 1]
                    weight = sign[a] * np.exp(1j * theta[guess, v, a])
                    block[:, :, v, int(t[a][v])] += weight * (Vv @ ra @ ra.conj().T).T
            if verify:
                _check_row_resolution(block)
        return A

    def close(self, state, sites, rng, verify):
        m = self.n_sectors
        sites, l_one, r_one = super().close(state, sites, rng, verify)
        chi = np.arange(m)
        l_vec = np.zeros(self.n_copies * state.Ds[0], dtype=complex)
        for b in range(self.n_copies):
            guess, g = divmod(b, m)
            # With one guess the sectors differ only by these weights (a Mode II convolution).
            sectors = chi if self.n_guesses == 1 else chi[guess : guess + 1]
            weight = np.sum(np.exp(1j * state.phi[sectors] - 2j * np.pi * sectors * g / m)) / m
            l_vec[b * state.Ds[0] : (b + 1) * state.Ds[0]] = weight * l_one
        return sites, l_vec, np.tile(r_one, self.n_copies)
