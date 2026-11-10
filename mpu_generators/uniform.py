# mpu_generators/uniform.py
r"""Translation-invariant certified random-MPU generator.

:class:`UniformMPUGenerator` draws one site tensor and reuses it at every
site (``_resample = False``); all modes of
:class:`~mpu_generators.base.MPUGenerator` thereby become translation
invariant with no other change (thesis ``rem:TI-specialization``). The
returned tensor list then contains the same array ``N`` times, and
``tensors[0]`` can be passed directly to :class:`TreeMPU.UniformMPU`.

Additional translation-invariant constructions
----------------------------------------------
The remaining methods return a single bulk tensor ``(A, l_vec, r_vec)``
(``A`` of shape ``(d, d, D, D)``) instead of a list. They are specific to
the translation-invariant case and are **not** constructed in the thesis
or in the tensor-first sampling paper; they are validated by the unit
tests only:

* :meth:`~UniformMPUGenerator.clock`,
  :meth:`~UniformMPUGenerator.group_automaton` and
  :meth:`~UniformMPUGenerator.degenerate_rows` are special cases of Mode I
  row resolution (``group_automaton("Zn", n=2)`` is the CNOT-staircase
  example of the paper, :math:`A[p,q]=P_qX^p`, :math:`\ell=e_0`);
  ``clock`` returns a chain of on-site unitaries that cycle with period
  :math:`D` along the chain, the other two are in general entangling;
* :meth:`~UniformMPUGenerator.shared_basis_automaton` and
  :meth:`~UniformMPUGenerator.cayley_automaton` are Mode I tensors in which
  every vertex reads the site in the *same* basis: a letter read advances a
  counter (the vertex) by a fixed map, and a unitary attached to the vertex
  acts on the site. They contain ``group_automaton`` (basis = computational,
  letters = all group elements) and Mode II ``cyclic`` (identity unitaries,
  nontrivial final phases) as special cases; the tail of operator spreading
  does not decay when the maps are permutations (see the method notes);
* :meth:`~UniformMPUGenerator.self_direction`,
  :meth:`~UniformMPUGenerator.resonant` and
  :meth:`~UniformMPUGenerator.multi_symbol` give
  :math:`U_N = c_N\,W^{\otimes N}`, an on-site product unitary with zero
  operator entanglement (useful as controls, not as entangling generators);
* :meth:`~UniformMPUGenerator.invariant_subspace` embeds a smaller
  translation-invariant MPU in a larger bond space.

Every method uses the package convention ``A[p_in, p_out, bond_L, bond_R]``.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from .base import (
    MPUGenerator,
    _as_unitary,
    _cayley_transitions,
    _check_row_resolution,
    _haar_unitary,
    _mode1_site_tensor,
    _random_out_regular_digraph,
    _shared_basis_site,
)

#: Scale of the random (uncertified) entries in
#: :meth:`UniformMPUGenerator.invariant_subspace`; any value works, a modest
#: one keeps the tensor well conditioned.
_JUNK_SCALE = 0.3
#: Frobenius-norm tolerance for accepting a user-supplied matrix as unitary.
_UNITARY_ATOL = 1e-9


def _rng(seed) -> np.random.Generator:
    r"""Return ``numpy.random.default_rng(seed)``."""
    return np.random.default_rng(seed)


class UniformMPUGenerator(MPUGenerator):
    r"""Translation-invariant variant of :class:`~mpu_generators.base.MPUGenerator`.

    One site tensor is drawn per call and reused at every site. With
    ``coherent_in_n=True`` (default) the site tensor is independent of
    ``n_sites`` for a fixed seed. See the module docstring for the additional
    single-tensor constructions.

    Parameters
    ----------
    seed : int
        Seed of the random number generator (default 0).
    verify : bool
        Run the algebraic checks at construction time and the numerical
        safeguards on every generated chain (default True); see
        :class:`~mpu_generators.base.MPUGenerator`.
    coherent_in_n : bool
        If True (default) the random stream does not depend on the chain
        length; see :class:`~mpu_generators.base.MPUGenerator`.

    Attributes
    ----------
    seed, verify, coherent_in_n
        As given.
    """

    _resample = False

    # --- special cases of Mode I row resolution ---

    def clock(
        self, D: int = 3, d: int = 2, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a :math:`D`-cycle tensor with Haar unitaries on its edges.

        Not part of the thesis or paper; a special case of Mode I row
        resolution with one target per row.

        :math:`A[a,a+1 \bmod D] = u_a` with independent :math:`u_a\sim\mathrm{Haar}(U(d))`.
        Every row has a single target, so (RR) reduces to :math:`u_a^\dagger u_a=1`.
        Boundary :math:`\ell=e_0`, :math:`r=(1,\dots,1)`; the chain is the
        on-site product :math:`u_0\otimes u_1\otimes\cdots` (cyclically, with
        period :math:`D`), which has operator-Schmidt rank 1.

        Parameters
        ----------
        D : int
            Bond dimension (cycle length).
        d : int
            Physical dimension.
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``A`` of shape ``(d, d, D, D)``; ``l_vec = e_0`` and
            ``r_vec = (1, ..., 1)``, both of length ``D``.
        """
        rng = _rng(self.seed if seed is None else seed)
        A = np.zeros((d, d, D, D), dtype=complex)
        for a in range(D):
            b = (a + 1) % D
            A[:, :, a, b] = _haar_unitary(d, rng).T
        l_vec = np.zeros(D, dtype=complex)
        l_vec[0] = 1.0
        r_vec = np.ones(D, dtype=complex)
        return A, l_vec, r_vec

    def group_automaton(
        self,
        group: str = "Zn",
        n: int = 3,
        cocycle=None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return the group-multiplication automaton of a finite group.

        The tensor is :math:`A[g,h]=|h\rangle\langle g^{-1}h|`.

        Not part of the thesis or paper (except the ``Zn``, ``n=2`` CNOT
        example); a special case of Mode I row resolution.

        For a finite group :math:`G` (:math:`D=d=|G|`), optionally twisted by a
        2-cocycle, :math:`A[g,h]=\omega(g,g^{-1}h)\,|h\rangle\langle g^{-1}h|`.
        (RR) holds for every group (permutation algebra; unimodular phases do
        not affect it) and is checked numerically. Boundary :math:`\ell=e_{\rm id}`,
        :math:`r=(1,\dots,1)`. For ``group="Zn", n=2`` this is the CNOT witness
        of the tensor-first sampling paper.

        Parameters
        ----------
        group : {"Zn", "Sn"}
            Cyclic group of order ``n`` or symmetric group on ``n`` letters.
        n : int
            Group parameter.
        cocycle : callable, optional
            ``cocycle(g, h) -> complex`` of unit modulus.
        seed : int, optional
            Unused (the construction is deterministic); kept for API symmetry.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``A`` of shape ``(d, d, D, D)`` with ``D = d = |G|``.

        Raises
        ------
        ValueError
            If ``group`` is not ``"Zn"`` or ``"Sn"``.

        Notes
        -----
        Combined multi-vertex readout is not unitary for this family: the
        cross terms satisfy :math:`\|Q_N[g,g']\|=|G|^{N/2}` for :math:`g\neq g'`.
        """
        if group == "Zn":
            elements = list(range(n))
            mult = lambda g, h: (g + h) % n
            inv = lambda g: (-g) % n
        elif group == "Sn":
            elements = list(permutations(range(n)))
            mult = lambda g, h: tuple(g[h[i]] for i in range(n))
            inv = lambda g: tuple(sorted(range(n), key=lambda i: g[i]))
        else:
            raise ValueError(f"Unknown group {group!r}; expected 'Zn' or 'Sn'.")

        D = d = len(elements)
        idx = {g: i for i, g in enumerate(elements)}
        identity = elements[0] if group == "Zn" else tuple(range(n))

        A = np.zeros((d, d, D, D), dtype=complex)
        for gi, g in enumerate(elements):
            g_inv = inv(g)
            for hi, h in enumerate(elements):
                target = mult(g_inv, h)
                phase = 1.0 if cocycle is None else cocycle(g, target)
                A[idx[target], idx[h], gi, hi] = phase

        l_vec = np.zeros(D, dtype=complex)
        l_vec[idx[identity]] = 1.0
        r_vec = np.ones(D, dtype=complex)
        return A, l_vec, r_vec

    # --- shared reading basis (counter automata) ---

    def shared_basis_automaton(
        self,
        transitions: list[list[int]] | None = None,
        D: int = 4,
        d: int = 2,
        R: np.ndarray | None = None,
        V: list[np.ndarray] | None = None,
        final_phases: np.ndarray | None = None,
        start: int = 0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a Mode I counter automaton that reads every site in one basis.

        Not part of the thesis or paper; a special case of Mode I row
        resolution (with a diagonal phase on the right boundary) in which the
        input-side projectors do not depend on the vertex.

        Let :math:`R=(r_0,\dots,r_{d-1})` be a unitary (the reading basis),
        :math:`V_v` a unitary attached to vertex :math:`v`, and
        :math:`\sigma_a:\{0..D-1\}\to\{0..D-1\}` the move caused by letter
        :math:`a`. The tensor is

        .. math::
            A[v,\sigma_a(v)] \mathrel{+}= V_v\,|r_a\rangle\langle r_a| ,

        with :math:`\ell=e_{\rm start}` and
        :math:`r_u=e^{i\varphi_u}`. On the basis state
        :math:`|r_{x_1}\cdots r_{x_N}\rangle` the chain acts as

        .. math::
            U_N\,|r_{x_1}\cdots r_{x_N}\rangle
            = e^{i\varphi_{v_N}}\,(V_{v_0}r_{x_1})\otimes\cdots\otimes(V_{v_{N-1}}r_{x_N}),
            \qquad v_j=\sigma_{x_j}(v_{j-1}),

        a single path with a unimodular amplitude per input basis vector.
        Two inputs that first differ at site :math:`j` have the same
        :math:`v_{j-1}` and are mapped to orthogonal vectors on site
        :math:`j`, so the images are orthonormal: :math:`U_N` is unitary for
        every :math:`N`, :math:`R`, :math:`V_v`, :math:`\varphi` and every
        choice of the maps :math:`\sigma_a` (not only permutations).

        Parameters
        ----------
        transitions : list of list of int, optional
            ``transitions[a][v]`` is :math:`\sigma_a(v)`, ``d`` lists of
            length ``D`` with entries in ``0..D-1``. Default: the lazy ring,
            :math:`\sigma_a(v)=v+a \bmod D`.
        D, d : int
            Bond and physical dimension (``D`` is taken from ``transitions``
            if given, so that ``d`` is then ``len(transitions)``).
        R : ndarray, optional
            Reading basis, a unitary of shape ``(d, d)`` (Haar if omitted).
        V : list of ndarray, optional
            ``D`` unitaries of shape ``(d, d)`` (independent Haar if omitted).
            With ``V = [identity] * D`` and non-trivial ``final_phases`` the
            chain is :math:`R\,e^{i\varphi(\text{counter})}R^\dagger`
            (Mode II cyclic for the ring).
        final_phases : ndarray, optional
            Phases :math:`\varphi_u`, length ``D`` (default zero, so that
            :math:`r=(1,\dots,1)`).
        start : int
            Initial vertex (:math:`\ell=e_{\rm start}`).
        seed : int, optional
            Overrides the generator seed for this call. Draw order: ``R``,
            then ``V[0], ..., V[D-1]``.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``A`` of shape ``(d, d, D, D)``; ``l_vec = e_start`` and
            ``r_vec = exp(1j * final_phases)``.

        Raises
        ------
        ValueError
            If ``transitions``, ``R``, ``V``, ``final_phases`` or ``start``
            have the wrong shape or range, or ``R`` / ``V`` are not unitary.

        Notes
        -----
        *Permutations versus general maps.* If every :math:`\sigma_a` is a
        permutation (a group automaton when they generate a group), two
        different paths never merge, and a change of one input letter changes
        the vertex of every later site; with :math:`V_v` distinct the
        operator :math:`U_N P_0U_N^\dagger` keeps a weight that does not
        decay with the distance, in one direction (towards larger site
        index). When the basis is instead drawn independently at each
        vertex, as by :meth:`~mpu_generators.base.MPUGenerator.branching`,
        the weight outside a window of radius :math:`R` decays with
        :math:`R` (numerically; no analytic rate is claimed here). This is a
        property of the operator, not of the cost of :mod:`TreeMPU`: the
        merge-tree synthesis depends only on :math:`\chi`, :math:`d` and the
        conditioning number.

        The chain is a one-directional counter automaton, so it is not
        invariant under reflection of the chain (unlike Mode II).

        ``D > d`` is allowed (numerically, the ring with :math:`D>d` has
        operator-Schmidt rank :math:`D` for long enough chains). If
        Assumption 1 fails at one site, block the tensor with
        :func:`TreeMPU.utils.block_mpu_tensors`, or use
        ``boundary_mode1_exact=True`` in :class:`TreeMPU.UniformMPU` when
        ``final_phases`` is zero.

        Examples
        --------
        >>> gen = UniformMPUGenerator(seed=0)
        >>> A, l_vec, r_vec = gen.shared_basis_automaton(D=4, d=2)
        >>> A.shape
        (2, 2, 4, 4)
        """
        rng = _rng(self.seed if seed is None else seed)
        if transitions is None:
            if D < 1 or d < 2:
                raise ValueError(f"need D >= 1 and d >= 2; got D={D}, d={d}.")
            transitions = [[(v + a) % D for v in range(D)] for a in range(d)]
        else:
            d = len(transitions)
            D = len(transitions[0]) if d else 0
            if d < 2 or D < 1 or any(len(t) != D for t in transitions):
                raise ValueError("transitions must be d >= 2 lists of the same length D >= 1.")
            if any(not (0 <= int(u) < D) for t in transitions for u in t):
                raise ValueError(f"transitions entries must lie in 0..{D - 1}.")
        if not 0 <= start < D:
            raise ValueError(f"start={start} out of range for D={D}.")

        R = _haar_unitary(d, rng) if R is None else _as_unitary(R, d, "R", _UNITARY_ATOL)
        if V is None:
            V = [_haar_unitary(d, rng) for _ in range(D)]
        else:
            if len(V) != D:
                raise ValueError(f"V must contain D={D} unitaries; got {len(V)}.")
            V = [_as_unitary(Vv, d, "V[v]", _UNITARY_ATOL) for Vv in V]
        phases = np.zeros(D) if final_phases is None else np.asarray(final_phases, dtype=float)
        if phases.shape != (D,):
            raise ValueError(f"final_phases must have shape ({D},); got {phases.shape}.")

        A = _shared_basis_site(transitions, R, V, d, D)
        _check_row_resolution(A)
        l_vec = np.zeros(D, dtype=complex)
        l_vec[start] = 1.0
        return A, l_vec, np.exp(1j * phases)

    def cayley_automaton(
        self,
        group: str = "S3",
        generators: list | None = None,
        group_n: int | list[int] = 0,
        R: np.ndarray | None = None,
        V: list[np.ndarray] | None = None,
        final_phases: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a shared-basis automaton whose vertices are the elements of a group.

        Not part of the thesis or paper. The letters are ``generators``
        (physical dimension :math:`d` = their number) and reading letter
        :math:`a` multiplies the vertex on the right, :math:`v\mapsto v\,g_a`;
        the start vertex is the identity. This is
        :meth:`shared_basis_automaton` on the Cayley graph, so
        :math:`D=|G|` can exceed :math:`d` (for instance :math:`S_3` with a
        transposition and a 3-cycle gives :math:`d=2`, :math:`D=6`).

        Parameters
        ----------
        group : {"S3", "Zn", "Zn_product"}
            Group, as in :meth:`~mpu_generators.base.MPUGenerator.fusion`.
        generators : list, optional
            Group elements (as listed by the group table: permutation tuples
            for ``S3``, integers for ``Zn``, tuples for ``Zn_product``). At
            least two are needed. Default for ``S3``: the first element of
            order 2 and the first of order 3. Required for the other groups.
        group_n : int or list of int
            Order of ``Zn`` or factor orders of ``Zn_product``.
        R, V, final_phases, seed
            As in :meth:`shared_basis_automaton`.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            As in :meth:`shared_basis_automaton`, with ``D = |G|``.

        Raises
        ------
        ValueError
            If ``generators`` is missing (for a group other than ``S3``), has
            fewer than two elements, or contains a non-element.

        Notes
        -----
        The generators are not required to generate the group; if they do
        not, the reachable vertices form a subgroup coset structure and the
        remaining vertices are unused. ``group_automaton`` is the case
        ``generators`` = all elements, :math:`R=1` and :math:`V_g` = left
        multiplication by :math:`g`.
        """
        transitions, start, _d = _cayley_transitions(group, generators, group_n)
        return self.shared_basis_automaton(
            transitions=transitions,
            R=R,
            V=V,
            final_phases=final_phases,
            start=start,
            seed=seed,
        )

    # --- single-tensor constructions (on-site products, embeddings) ---

    def self_direction(
        self,
        D: int = 4,
        d: int = 2,
        mu: complex = 1.0,
        W: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a tensor with :math:`\sum_b A[a,b]\,r_b = \mu\, r_a W`.

        Not part of the thesis or paper. The chain is an on-site product
        unitary (see Notes).

        With :math:`|\mu|=1` and :math:`W` unitary, induction gives
        :math:`U_N=\mu^N\langle\bar\ell, r\rangle\,W^{\otimes N}`, unitary for
        :math:`\langle\bar\ell,r\rangle=1`. The tensor is
        :math:`A[a,b]=\mu\,r_a\bar r_b W/\|r\|^2 + \sum_c C[a,c]P_{cb}` with
        :math:`P=1-rr^\dagger/\|r\|^2` and random :math:`C` (the second term is
        annihilated by :math:`r`).

        Parameters
        ----------
        D, d : int
            Bond and physical dimension.
        mu : complex
            Unimodular eigenvalue.
        W : ndarray, optional
            Physical unitary (Haar if omitted).
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``l_vec = conj(r)/|r|^2`` (boundary vectors pair linearly with the
            bond index).

        Notes
        -----
        :math:`U_N` is an on-site product unitary (zero operator entanglement);
        its unitarity is checked numerically.
        """
        rng = _rng(self.seed if seed is None else seed)
        r_vec = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        norm2 = float(np.vdot(r_vec, r_vec).real)
        if W is None:
            W = _haar_unitary(d, rng)
        P = np.eye(D, dtype=complex) - np.outer(r_vec, r_vec.conj()) / norm2
        C = rng.standard_normal((D, D, d, d)) + 1j * rng.standard_normal((D, D, d, d))
        junk = np.einsum("acij,cb->abij", C, P)
        # Assemble as [bond_L, bond_R, p_out, p_in], then move to package order.
        A_bond_first = (
            mu * np.outer(r_vec, r_vec.conj())[:, :, None, None] / norm2 * W[None, None, :, :]
            + junk
        )
        A = np.ascontiguousarray(A_bond_first.transpose(3, 2, 0, 1))
        # Boundary vectors pair linearly with the bond index, so conj(l) . r = 1
        # requires l_vec = conj(r) / |r|^2.
        l_vec = (r_vec.conj() / norm2).astype(complex)
        return A, l_vec, r_vec.astype(complex)

    def resonant(
        self,
        D: int = 5,
        d: int = 2,
        target_phases: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a tensor whose bond matrix has a root-of-unity spectrum.

        The spectrum is the set of :math:`D`-th roots of unity. Not part of
        the thesis or paper. The chain is an on-site product unitary times a
        phase.

        :math:`A[a,b]=M[a,b]\,\tilde W` with :math:`M=S\,\mathrm{diag}(\omega^k)\,S^{-1}`,
        :math:`\omega=e^{2\pi i/D}`, :math:`S` a random invertible (non-unitary)
        matrix and :math:`\tilde W` Haar. Then :math:`U_N=c_N\,\tilde W^{\otimes N}`
        with :math:`c_N=\bar\ell^\top M^N r`, and :math:`\bar\ell` is solved so that
        :math:`c_N` equals the prescribed unimodular phases for
        :math:`N=0,\dots,D-1` (hence for all :math:`N`, since :math:`M^D=1`).

        Parameters
        ----------
        D, d : int
            Bond and physical dimension.
        target_phases : ndarray, optional
            :math:`(c_0,\dots,c_{D-1})`, unimodular (random if omitted).
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``A`` of shape ``(d, d, D, D)``; ``l_vec`` and ``r_vec`` of length
            ``D``.

        Notes
        -----
        :math:`U_N` is an on-site product unitary times a phase.
        """
        rng = _rng(self.seed if seed is None else seed)
        omega = np.exp(2j * np.pi / D)
        Lam = np.diag(omega ** np.arange(D))
        S = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        M = S @ Lam @ np.linalg.inv(S)
        W_tilde = _haar_unitary(d, rng)
        r_vec = rng.standard_normal(D) + 1j * rng.standard_normal(D)

        if target_phases is None:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=D)
            target_phases = np.exp(1j * theta)
        target_phases = np.asarray(target_phases, dtype=complex)

        # Solve conj(l)^T M^N r = c_N for N = 0..D-1 (linear in conj(l)).
        Mr = r_vec.copy()
        rows = []
        for _ in range(D):
            rows.append(Mr.copy())
            Mr = M @ Mr
        Rmat = np.stack(rows, axis=0)  # (D, D): row N is (M^N r)
        l_bar = np.linalg.solve(Rmat, target_phases)  # this IS conj(l)

        # Assemble as [bond_L, bond_R, p_out, p_in], then move to package order.
        A_bond_first = M[:, :, None, None] * W_tilde[None, None, :, :]
        A = np.ascontiguousarray(A_bond_first.transpose(3, 2, 0, 1))
        # l_bar already is the linearly-paired boundary vector.
        return A, l_bar.astype(complex), r_vec.astype(complex)

    def multi_symbol(
        self, d: int = 3, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return :meth:`self_direction` with a diagonal :math:`W`.

        Not part of the thesis or paper.

        :math:`W=\mathrm{diag}(e^{i\theta_0},\dots,e^{i\theta_{d-1}})`, :math:`\mu=1`,
        :math:`D=d`; :math:`U_N` applies a phase depending only on how many times
        each physical symbol occurs; the chain is an on-site product unitary
        :math:`W^{\otimes N}`.

        Parameters
        ----------
        d : int
            Physical dimension (the bond dimension equals ``d``).
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``A`` of shape ``(d, d, d, d)``; ``l_vec`` and ``r_vec`` of
            length ``d``.
        """
        rng = _rng(self.seed if seed is None else seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        D_mu = np.diag(np.exp(1j * theta)).astype(complex)
        return self.self_direction(D=d, d=d, mu=1.0, W=D_mu, seed=seed)

    def invariant_subspace(
        self,
        sub_A: np.ndarray,
        sub_l: np.ndarray,
        sub_r: np.ndarray,
        D_extra: int = 2,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Embed a translation-invariant MPU in a larger bond space.

        Not part of the thesis or paper.

        In a random (non-axis-aligned) basis :math:`S=[F\,|\,G]`, the bond matrix
        is block upper-triangular: ``sub_A`` on the :math:`k`-dimensional
        subspace :math:`\mathcal R=\mathrm{span}(F)`, zero leakage out of
        :math:`\mathcal R`, random entries elsewhere. Boundaries are supported on
        :math:`\mathcal R`, so :math:`U_N` equals the chain of ``sub_A`` for every
        :math:`N`.

        Parameters
        ----------
        sub_A, sub_l, sub_r : ndarray
            A valid translation-invariant MPU of bond dimension :math:`k`.
        D_extra : int
            Additional bond dimension (total :math:`D=k+D_{\rm extra}`).
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``l_vec = (S^T)^{-1} l_sub``, ``r_vec = S r_sub``.
        """
        d = sub_A.shape[0]
        k = sub_A.shape[2]
        D = k + D_extra
        rng = _rng(self.seed if seed is None else seed)

        S = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        S_inv = np.linalg.inv(S)

        A_tilde = rng.standard_normal((d, d, D, D)) + 1j * rng.standard_normal((d, d, D, D))
        A_tilde *= _JUNK_SCALE
        A_tilde[:, :, :k, :k] = sub_A
        A_tilde[:, :, k:, :k] = 0.0  # no leakage out of R when starting inside R

        # Bond axes only (positions 2,3) get conjugated by S; physical axes
        # (0,1) pass through unchanged, since d is shared with the full system.
        A = np.einsum("aA,xyAB,Bb->xyab", S, A_tilde, S_inv)

        l_tilde = np.zeros(D, dtype=complex)
        l_tilde[:k] = sub_l
        r_tilde = np.zeros(D, dtype=complex)
        r_tilde[:k] = sub_r
        # Linear pairing: under v = S v', l -> (S^T)^{-1} l and r -> S r.
        l_vec = np.linalg.inv(S.T) @ l_tilde
        r_vec = S @ r_tilde
        return A, l_vec.astype(complex), r_vec.astype(complex)

    def degenerate_rows(
        self,
        D: int = 2,
        d: int = 2,
        groups: list[list[int]] | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return a Mode I tensor with literally identical rows.

        Not part of the thesis or paper beyond the remark cited below.

        Starting from a Mode I row-resolution tensor, the rows in each group are
        overwritten with copies of the group's first row. (RR) is a per-row
        condition, so it still holds, and the partial chains
        :math:`\Xi_n[a]=\sum_b A[a,b]\otimes\Xi_{n-1}[b]` coincide for all rows of
        a group at every :math:`n` (thesis ``rem:row-duplication``).

        Parameters
        ----------
        D, d : int
            Bond and physical dimension, ``D >= 2``.
        groups : list of list of int, optional
            Disjoint groups of row indices to make identical (default
            ``[[0, 1]]``).
        seed : int, optional
            Overrides the generator seed for this call.

        Returns
        -------
        A, l_vec, r_vec : ndarray
            ``l_vec = e_{groups[0][0]}``, ``r_vec = (1, ..., 1)``.

        Raises
        ------
        ValueError
            If ``D < 2``, a group has fewer than two rows, a row index is out
            of range, or a row appears in more than one group.
        """
        if D < 2:
            raise ValueError(f"degenerate_rows needs D >= 2, got D={D}.")
        if groups is None:
            groups = [[0, 1]]
        seen = set()
        for g in groups:
            if len(g) < 2:
                raise ValueError(f"each group needs >= 2 rows to be degenerate; got {g}.")
            for a in g:
                if not (0 <= a < D):
                    raise ValueError(f"row index {a} out of range for D={D}.")
                if a in seen:
                    raise ValueError(f"row {a} appears in more than one group.")
                seen.add(a)

        rng = _rng(self.seed if seed is None else seed)
        edges = _random_out_regular_digraph(D, d, rng)
        A = _mode1_site_tensor(D, edges, d, rng)
        for g in groups:
            src = g[0]
            for a in g[1:]:
                A[:, :, a, :] = A[:, :, src, :]
        _check_row_resolution(A)  # (RR) must hold at every row, duplicated ones included

        l_vec = np.zeros(D, dtype=complex)
        l_vec[groups[0][0]] = 1.0
        r_vec = np.ones(D, dtype=complex)
        return A, l_vec, r_vec
