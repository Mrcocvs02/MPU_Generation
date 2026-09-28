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
instead of a list. They come from earlier working notes on the
translation-invariant case and are **not** part of the thesis chapter or
the tensor-first sampling paper:

* :meth:`~UniformMPUGenerator.clock`,
  :meth:`~UniformMPUGenerator.group_automaton` and
  :meth:`~UniformMPUGenerator.degenerate_rows` are special cases of Mode I
  row resolution (``group_automaton("Zn", n=2)`` is the CNOT-staircase
  witness of the paper, :math:`A[p,q]=P_qX^p`, :math:`\ell=e_0`);
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
    _check_row_resolution,
    _haar_unitary,
    _mode1_site_tensor,
    _random_out_regular_digraph,
)

#: Scale of the random (uncertified) entries in :meth:`UniformMPUGenerator.invariant_subspace`;
#: any value works, a modest one keeps the tensor well conditioned.
_JUNK_SCALE = 0.3


def _rng(seed) -> np.random.Generator:
    r"""Return ``numpy.random.default_rng(seed)``."""
    return np.random.default_rng(seed)


class UniformMPUGenerator(MPUGenerator):
    r"""Translation-invariant variant of :class:`~mpu_generators.base.MPUGenerator`.

    One site tensor is drawn per call and reused at every site, so the site
    tensor is independent of ``n_sites`` for a fixed seed. See the module
    docstring for the additional single-tensor constructions.
    """

    _resample = False

    # =================================================================
    # Special cases of Mode I row resolution
    # =================================================================

    def clock(
        self, D: int = 3, d: int = 2, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Clock construction: a :math:`D`-cycle with Haar unitaries on its edges.

        :math:`A[a,a+1 \bmod D] = u_a` with independent :math:`u_a\sim\mathrm{Haar}(U(d))`.
        Every row has a single target, so (RR) reduces to :math:`u_a^\dagger u_a=1`.
        Boundary :math:`\ell=e_0`, :math:`r=(1,\dots,1)`; the chain is the
        on-site product :math:`u_0\otimes u_1\otimes\cdots` (cyclically).

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
        r"""Group-multiplication automaton, :math:`A[g,h]=|h\rangle\langle g^{-1}h|`.

        For a finite group :math:`G` (:math:`D=d=|G|`), optionally twisted by a
        2-cocycle, :math:`A[g,h]=\omega(g,g^{-1}h)\,|h\rangle\langle g^{-1}h|`.
        (RR) holds exactly for every group (permutation algebra; unimodular
        phases do not affect it). Boundary :math:`\ell=e_{\rm id}`,
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

    # =================================================================
    # Single-tensor constructions (on-site products, embeddings)
    # =================================================================

    def self_direction(
        self,
        D: int = 4,
        d: int = 2,
        mu: complex = 1.0,
        W: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Self-direction construction: :math:`\sum_b A[a,b]\,r_b = \mu\, r_a W`.

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
        :math:`U_N` is an on-site product unitary (zero operator entanglement).
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
        r"""Resonant construction: a bond matrix with spectrum on the :math:`D`-th roots of unity.

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
        r"""Multi-symbol phase automaton: :meth:`self_direction` with a diagonal :math:`W`.

        :math:`W=\mathrm{diag}(e^{i\theta_0},\dots,e^{i\theta_{d-1}})`, :math:`\mu=1`,
        :math:`D=d`; :math:`U_N` applies a phase depending only on how many times
        each physical symbol occurs.

        Returns
        -------
        A, l_vec, r_vec : ndarray
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
        r"""Mode I tensor with literally identical rows.

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
