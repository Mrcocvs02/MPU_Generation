# mpu_generators/uniform.py
r"""
Translationally-invariant (TI) certified random-MPU generator.

``UniformMPUGenerator`` overrides exactly the resample-vs-reuse decision
in :class:`mpu_generators.base.MPUGenerator._sites` (``_resample = False``),
which alone turns every one of the base class's four modes into its
TI-constrained counterpart -- [PDF-B] Part I: "vincolare al caso TI" is
literally "draw once instead of per site", nothing else changes for
``branching``/``fusion``/``fusion_bond_resolved``/``grafted``.

Everything below this line is genuinely TI-only content from [PDF-B] Parts
II-IV: mechanisms that only make sense for a *single reused tensor* applied
recursively (self-direction, invariant subspaces, ...), not restrictions of
a per-site-resampled construction. Each method returns ``(A, l_vec, r_vec)``
for the single reused bulk tensor, in this package's own convention
``(p_in, p_out, bond_L, bond_R)``, ready for ``UniformMPU(A, l_vec, r_vec, N=...)``.

Deliberately **not implemented** (both source documents show these fail,
[PDF-B] §2.4/§3.7): ancilla-dilation relaxation of (RR) (dressing a genuine
unitary on ancilla(D) (x) physical(d) does not give (RR) -- the two
resolutions of identity it does give sum over the wrong index), and the
triangular/Jordan extension (a "falling" branch leaves an irreducible
boundary mismatch with the surrounding bulk operator, independent of
whether the two bulk unitaries coincide). No code for either is provided.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from .base import (
    MPUGenerator,
    _haar_unitary,
    _pdf_to_package_tensor,
    _random_out_regular_digraph,
    _mode1_site_tensor,
    _check_row_resolution,
)


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


class UniformMPUGenerator(MPUGenerator):
    """
    Translationally-invariant variant: every mode draws its per-site
    dressing once and reuses it at every site (see module docstring).
    """

    _resample = False

    # =================================================================
    # Part II: named ansatz families ([PDF-B] enrichments of Mode I / II-B,
    # not independent mechanisms -- kept here as convenient constructors).
    # =================================================================

    def clock(self, D: int = 3, d: int = 2, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Clock / periodic dressing ([PDF-B] §2.1): a $D$-cycle $\sigma$ and
        $D$ independent Haar unitaries $u_0,\dots,u_{D-1}$,
        $A[a,\sigma(a)]=u_a$. A degree-1 special case of Mode I row
        resolution (RR) holds trivially: each row has a single target, so
        completeness is just $u_a^\dagger u_a = 1_d$ (Haar-unitary), and
        there is no cross term to cancel. Single-vertex readout,
        $l=e_0$, $r=\sum_v e_v$.
        """
        rng = _rng(self.seed if seed is None else seed)
        A_pdf = np.zeros((D, D, d, d), dtype=complex)
        for a in range(D):
            b = (a + 1) % D
            A_pdf[a, b] = _haar_unitary(d, rng)
        l_pdf = np.zeros(D, dtype=complex)
        l_pdf[0] = 1.0
        r_pdf = np.ones(D, dtype=complex)
        return _pdf_to_package_tensor(A_pdf), l_pdf, r_pdf

    def group_automaton(
        self,
        group: str = "Zn",
        n: int = 3,
        cocycle=None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Group-multiplication automaton ([PDF-B] §2.2): $A[g,h]=|h\rangle
        \langle g^{-1}h|$ for any finite group $G$ ($D=d=|G|$), optionally
        dressed by a $2$-cocycle $\omega \in H^2(G,U(1))$,
        $A[g,h] = \omega(g,g^{-1}h)|h\rangle\langle g^{-1}h|$ (unimodular
        phases never break the orthogonality argument, so (RR) still holds
        exactly). ``group``: ``"Zn"`` (cyclic, ``n`` elements) or ``"Sn"``
        (symmetric group on ``n`` letters, ``n!`` elements).

        (RR) holds *exactly* for any finite group by pure permutation
        algebra (verified: zero residual, not just small, in
        ``tests/``) -- but combined multi-branch readout is **not** free
        here: $\|Q_N[g,g']\| = \sqrt{|G|^N}$ exactly for $g \neq g'$ (also
        verified as a regression, reproducing [PDF-B]'s own negative
        result), unlike the branch-absence trick the CNOT-staircase
        instance relies on.
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

        A_pdf = np.zeros((D, D, d, d), dtype=complex)
        for gi, g in enumerate(elements):
            g_inv = inv(g)
            for hi, h in enumerate(elements):
                target = mult(g_inv, h)
                phase = 1.0 if cocycle is None else cocycle(g, target)
                A_pdf[gi, hi, idx[h], idx[target]] = phase

        l_pdf = np.zeros(D, dtype=complex)
        l_pdf[idx[identity]] = 1.0
        r_pdf = np.ones(D, dtype=complex)
        return _pdf_to_package_tensor(A_pdf), l_pdf, r_pdf

    # =================================================================
    # Part III: genuinely new, TI-only mechanisms.
    # =================================================================

    def self_direction(
        self,
        D: int = 4,
        d: int = 2,
        mu: complex = 1.0,
        W: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Self-direction ansatz $(\star)$ ([PDF-B] §3.2):
        $\sum_b A[a,b] r_b = \mu r_a W$, $|\mu|=1$, $W$ unitary.

        Constructive recipe: $A[a,b] = \mu\frac{r_a\bar r_b}{\|r\|^2}W +
        \sum_c C[a,c] P_{cb}$, $P = 1_D - rr^\dagger/\|r\|^2$ (projector onto
        $r^\perp$), $C$ arbitrary "junk" -- verified algebraically that the
        junk term vanishes exactly against $(\star)$'s defining equation
        (since $Pr=0$), independent of what $C$ is; $C$ is drawn Haar-like
        random here purely to genuinely randomize the non-$r$ directions.

        By Proposition 4, $\Xi_n[a] = \mu^n r_a W^{\otimes n}$, so
        $U_N = \mu^N \langle l|r\rangle W^{\otimes N}$; the simplest boundary
        with $|\langle l|r\rangle|=1$ exactly is $l = r/\|r\|^2$, used here.
        """
        rng = _rng(self.seed if seed is None else seed)
        r_vec = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        norm2 = float(np.vdot(r_vec, r_vec).real)
        if W is None:
            W = _haar_unitary(d, rng)
        P = np.eye(D, dtype=complex) - np.outer(r_vec, r_vec.conj()) / norm2
        C = rng.standard_normal((D, D, d, d)) + 1j * rng.standard_normal((D, D, d, d))
        junk = np.einsum("acij,cb->abij", C, P)
        A_pdf = (
            mu * np.outer(r_vec, r_vec.conj())[:, :, None, None] / norm2 * W[None, None, :, :]
            + junk
        )
        # This module's l_vec/r_vec, as consumed by
        # compute_chain_boundary_operators/dense_chain, pair as U_N = sum_a
        # l_vec[a] Xi_N[a] -- i.e. l_vec IS ALREADY bar-l, the conjugated
        # object from PDF-B's own U_N = sum_a bar(l_a) Xi_N[a] notation.
        # r/norm2 solves bar(l).r = 1 for l = r/norm2 (since bar(r/norm2).r
        # = |r|^2/norm2 = 1), so the conjugated object to pass here is
        # conj(r/norm2) = conj(r)/norm2 -- verified directly against an
        # independent raw-recursion (Xi-based) contraction, not assumed.
        l_pdf = (r_vec.conj() / norm2).astype(complex)
        r_pdf = r_vec.astype(complex)
        return _pdf_to_package_tensor(A_pdf), l_pdf, r_pdf

    def resonant(
        self,
        D: int = 5,
        d: int = 2,
        target_phases: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Resonant generalization ([PDF-B] §3.4): spectrum on the $D$-th
        roots of unity ($M^D = 1$) instead of a single eigenvalue, reaching
        *any* target sequence of unimodular phases $(c_0,\dots,c_{D-1})$,
        solved linearly, with $M$ itself **not required to be unitary**.

        Own construction (independently re-derived and verified here, not a
        literal transcription of [PDF-B]'s compressed formula -- see
        ``CLAUDE.md``-equivalent notes in the implementation plan): embed
        $M = S\,\mathrm{diag}(\omega^0,\dots,\omega^{D-1})\,S^{-1}$
        ($\omega = e^{2\pi i/D}$, $S$ a random **non-unitary** invertible
        matrix, so $M$ is diagonalizable with the right spectrum but
        generically not unitary) directly into the bond structure,
        $A[a,b] := M[a,b]\,\tilde W$ for a single fixed physical unitary
        $\tilde W$. By induction $\Xi_N[a] = (M^N r)_a\,\tilde W^{\otimes N}$
        for any $r$ (algebra: $\Xi_1[a]=\sum_b A[a,b]r_b=(Mr)_a\tilde W$;
        $\Xi_{N+1}[a]=\sum_b A[a,b]\otimes\Xi_N[b]=\sum_b M[a,b]\tilde W
        \otimes(M^Nr)_b\tilde W^{\otimes N}=(M^{N+1}r)_a\tilde W^{\otimes(N+1)}$),
        so $U_N=\big(\sum_a\bar l_a(M^Nr)_a\big)\tilde W^{\otimes N}=c_N
        \tilde W^{\otimes N}$ with $c_N:=\bar l^T M^N r$ -- exactly unitary
        for *any* choice of $l$ with $|c_N|=1$, since $\tilde W$ is unitary
        by construction (not merely empirically): this makes the target
        sequence's exact reproduction and $M$'s non-unitarity provable
        by the same algebra, not just checked after the fact.
        ``target_phases``: length-``D`` array of unimodular target values
        $(c_0,\dots,c_{D-1})$ (random if not given); solved linearly for
        $l$ via $M^N r$, $N=0,\dots,D-1$.
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

        A_pdf = M[:, :, None, None] * W_tilde[None, None, :, :]
        r_pdf = r_vec.astype(complex)
        # As in self_direction: compute_chain_boundary_operators/dense_chain
        # want bar(l) directly (U_N = sum_a l_vec[a] Xi_N[a]), which is
        # exactly l_bar as solved above -- no further conjugation.
        return _pdf_to_package_tensor(A_pdf), l_bar.astype(complex), r_pdf

    def multi_symbol(
        self, d: int = 3, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Multi-symbol automaton ([PDF-B] §3.6). **Not an independent
        mechanism** -- [PDF-B] explicitly retracts its earlier
        classification as a fourth mechanism: it is exactly $(\star)$ with
        $\mu=1$, $W=\mathrm{diag}(\mu_0,\dots,\mu_{d-1})$ (verified there:
        $\Xi_1[a]=\sum_b A[a,b]r_b = r_a \cdot D_\mu$ directly). Implemented
        by calling :meth:`self_direction` with that ``W`` -- not
        reimplemented separately, matching the doc's own correction.
        Gives a phase depending on the *count* of each physical symbol
        along the chain, never on their order (a genuine, useful
        constructive recipe, just not a separate axis of the taxonomy).
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
        r"""
        Rank-$k$ invariant subspace ([PDF-B] §3.3): the reused tensor,
        restricted (in a **generic, non-axis-aligned basis**) to a
        $k$-dimensional subspace $\mathcal R \ni r$, must itself be a valid
        smaller TI-MPU tensor (``sub_A``, ``sub_l``, ``sub_r`` -- built by
        any other mechanism here, e.g. :meth:`self_direction`); the rest of
        the $D \times D$ matrix ($D = k + D\_extra$) is free, subject only
        to no leakage *out of* $\mathcal R$ (closure: $\hat A(\mathcal R)
        \subseteq \mathcal R \otimes M_d(\mathbb C)$).

        Construction: extend a random (non-axis-aligned) basis
        $F$ (columns spanning $\mathcal R$) to a full invertible $S=[F|G]$;
        in the transformed basis the tensor is block upper-triangular
        (``sub_A`` in the top-left $k\times k$ block, the bottom-left
        $(D-k)\times k$ block forced to zero -- no leakage into the junk
        rows from an $\mathcal R$ column -- top-right and bottom-right
        blocks free/random), then conjugated back by $S$. ``r``, ``l`` are
        supported only on $\mathcal R$'s coordinates in the transformed
        basis, i.e. $r = F r_{sub}$.
        """
        # sub_A/sub_l/sub_r come straight from another method here (e.g.
        # self_direction), already in this package's own convention
        # (p_in, p_out, bond_L, bond_R) -- worked entirely in that
        # convention below (bond axes last), not the PDF/reference one, so
        # no _pdf_to_package_tensor conversion is needed on the way out.
        d = sub_A.shape[0]
        k = sub_A.shape[2]
        D = k + D_extra
        rng = _rng(self.seed if seed is None else seed)

        S = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        S_inv = np.linalg.inv(S)

        A_tilde = rng.standard_normal((d, d, D, D)) + 1j * rng.standard_normal((d, d, D, D))
        A_tilde *= 0.3  # keep junk modest relative to the certified sub-block
        A_tilde[:, :, :k, :k] = sub_A
        A_tilde[:, :, k:, :k] = 0.0  # no leakage out of R when starting inside R

        # Bond axes only (positions 2,3) get conjugated by S; physical axes
        # (0,1) pass through unchanged, since d is shared with the full system.
        A_pkg = np.einsum("aA,xyAB,Bb->xyab", S, A_tilde, S_inv)

        l_tilde = np.zeros(D, dtype=complex)
        l_tilde[:k] = sub_l
        r_tilde = np.zeros(D, dtype=complex)
        r_tilde[:k] = sub_r
        # l_vec/r_vec pair *linearly* (no extra conjugate) with the bond
        # index in this package's own convention (verified directly for
        # self_direction above); the same v = S v' change of basis then
        # gives l_pdf = (S^T)^{-1} l_tilde, r_pdf = S r_tilde.
        l_pdf = np.linalg.inv(S.T) @ l_tilde
        r_pdf = S @ r_tilde
        return A_pkg, l_pdf.astype(complex), r_pdf.astype(complex)

    def degenerate_rows(
        self,
        D: int = 2,
        d: int = 2,
        groups: list[list[int]] | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Degenerate-rows counterexample family ([PDF-B] §3.5): literally
        identical rows of $A$ force $\Xi_n[a]=\Xi_n[a']$, *without* any
        product structure ($A \neq M \otimes \mathfrak u$) -- a genuinely
        different mechanism from $(\star)$ giving robust combined readout.

        **General construction (Task 5a: extends [PDF-B]'s hand-worked
        $D=d=2$ example to any $D \geq 2$, $d \geq 1$, and any grouping of
        rows, not just a single pair)**: $\Xi_{n+1}[a] = \sum_b A[a,b]
        \otimes \Xi_n[b]$ depends on $\Xi_n$ *only* through row $a$ of
        $A$ -- so if two rows are made literally identical as arrays
        (not just "linearly dependent"), $\Xi_n[a]=\Xi_n[a']$ holds at
        *every* $n \geq 1$ directly (no induction on $\Xi_n$ needed, since
        it never depends on which row produced it). This is true for
        *any* $A$, so starting from an ordinary Mode I row-resolution
        tensor (:meth:`branching`'s own construction, (RR) certified by
        Theorem 1 regardless of which rows happen to coincide -- (RR) is a
        per-row condition, duplicating a row's values onto another row
        cannot break it) and literally copying one row's blocks onto every
        other row in its group gives a family that is simultaneously (a)
        exactly unitary for single-vertex readout at any row in a group
        (ordinary Mode I), and (b) has $\Xi_n$ identical across every row
        within a group, for any $D$, $d$, and any partition of rows into
        groups -- [PDF-B]'s $D=d=2$, one-pair example is the smallest
        case of this, not a separate construction to generalize *from*.

        ``groups``: list of row-index lists to make mutually identical
        (default ``[[0, 1]]`` if ``D >= 2``); indices ``0..D-1``, no
        overlaps. ``l_vec = e_{g[0]}`` for the first group's first row
        (arbitrary choice -- $\Xi_n$ is identical across the whole group
        by construction, so which member is read out does not matter).

        **Open, not claimed here**: whether this family plus $(\star)$
        exhausts every construction with the robust-readout property is an
        explicitly open question in [PDF-B] -- generalizing the
        *construction* (this method) says nothing new about that
        classification question, which remains open.
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
        A_pdf = _mode1_site_tensor(D, edges, d, rng)
        for g in groups:
            src = g[0]
            for a in g[1:]:
                A_pdf[a] = A_pdf[src]
        _check_row_resolution(A_pdf)  # (RR) must hold at every row, duplicated ones included

        l_pdf = np.zeros(D, dtype=complex)
        l_pdf[groups[0][0]] = 1.0
        r_pdf = np.ones(D, dtype=complex)
        return _pdf_to_package_tensor(A_pdf), l_pdf, r_pdf
