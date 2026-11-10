# mpu_generators

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2508.08160-b31b1b.svg)](https://arxiv.org/abs/2508.08160)

Certified random generators for Matrix Product Unitaries (MPUs): sample
tensors that are unitary for every chain length $N$ under a
single fixed generative rule, rather than approximately unitary or
unitary only on average. Every construction except Mode II-B confines the
relevant transfer object to an $N$-independent algebraic family *by design*:
an identity of the local tensor and boundary data implies unitarity for every
length — nothing here relies on statistical typicality, large-$N$
concentration, or rejection sampling. Mode II-B (Lee–Yang) is validated
numerically only (see [Background](#background)).

**Which generator for what?** See [Choosing a generator](docs/userguide.md#4-choosing-a-generator): a table by goal, what keeps or destroys long-range operator spreading, and what Mode I cannot represent.

**Documentation:** [user guide](docs/userguide.md) ·
[tutorial notebooks](https://github.com/Mrcocvs02/TreeMPU/tree/main/notebooks) (02, 03, 06) ·
[theory → code map](https://github.com/Mrcocvs02/TreeMPU/blob/main/docs/theory_code_map.md)

Two classes, sharing one construction:

- `MPUGenerator` — general (non-uniform), site-resampled MPUs: a fresh
  Haar-random dressing is drawn independently at every physical site.
- `UniformMPUGenerator(MPUGenerator)` — translationally invariant MPUs
  (a single repeating bulk tensor), obtained "for free" from the same
  code by drawing that dressing once and reusing it.

## Contents

- [Key features](#key-features)
- [Background](#background)
- [Install](#install)
- [Quickstart](#quickstart)
- [API overview](#api-overview)
- [Mixing combinators](#mixing-combinators)
- [Verification and certification](#verification-and-certification)
- [Example: Mode II-B / Lee-Yang](#example-mode-ii-b--lee-yang)
- [Relationship to TreeMPU](#relationship-to-treempu)
- [Testing](#testing)
- [Reference](#reference)
- [Status](#status)
- [Roadmap](#roadmap)
- [AI usage disclosure](#ai-usage-disclosure)
- [License](#license)

## Key features

- **Mode I — `branching`** (row resolution): a digraph on the bond
  dimension whose rows are filled by Haar-dressed orthogonal blocks;
  completeness holds for *any* Haar draw, no rejection needed. The digraph
  and the rank partition are random and balanced by default, or
  hand-picked / random (`digraph`, `out_degree`, `ranks`). Supports
  combined multi-vertex readout via an auxiliary cascade of sites.
- **Mode II — `fusion`** (fusion closure): a bond-diagonal tensor whose
  channel matrices close under a fixed fusion rule. Four built-in
  skeletons — `cyclic` ($\mathbb{Z}_M$ characters; any $M$, also $M>d$; a
  group convolution only when $M$ divides $d$), `nested_flag` (shrinking-flag projectors, the
  diagonal-phase/MCZ-type family), `matrix_units` (identity channel + matrix
  units), and `group` (regular representation of a finite group, abelian or
  not; the construction applies to any finite group given its tables, and the
  implemented groups are `Zn`, `Zn_product` and `S3`).
- **Mode II-B — `fusion_bond_resolved`** (bond-resolved fusion): per-channel
  core tensors fixed once in a `FusionSkeleton`, then only the per-site Haar
  dressing is resampled. Ships with the Lee-Yang/Fibonacci skeleton
  (`lee_yang_skeleton`, $d=5$, bond dimension 14), whose tensors are typed
  explicitly. No algebraic certificate is checked at generation; unitarity is
  validated numerically only.
- **Grafted composition** (thesis `thm:grafted-factorization`): a Mode I
  two-vertex path algebra grafted inside the channels of the Mode II `cyclic`
  skeleton (a group convolution when the channel count divides $d$) — a
  product of two independently certified unitaries.
- **Shared-basis counter automata — `shared_basis_chain`, `cayley_chain`**
  (**new, beyond the thesis**): a family of Mode I tensors in which all the
  vertices of a site read the input in *one* orthonormal basis (the basis may
  differ from site to site), so the chain is a deterministic counter on the
  input letters: $U_N|r_{x_1}\cdots r_{x_N}\rangle=e^{i\varphi}\bigotimes_k
  V_{k,v_{k-1}}r_{k,x_k}$ with $v_k=\sigma_{k,x_k}(v_{k-1})$. Exactly unitary for
  every $N$, every basis, vertex unitary, phase, per-site bond dimension and
  map (a one-paragraph single-path proof, in the docstring and re-checked by
  (RR) on every tensor). They are notable because the operator spreading is
  one-directional and, for permutation maps, **does not decay with the
  distance** (a Haar-basis `branching` chain with the same digraph decays);
  they reach bond dimension $D>d$ (rings, Cayley graphs such as $S_3$ with
  $d=2$, $D=6$), contain `group_automaton` and Mode II `cyclic` as special
  cases, and `TreeMPU` synthesizes them exactly. `MPUGenerator` draws every
  site independently; `UniformMPUGenerator` reuses one site. Four named
  families: `shared_basis_ring`, `shared_basis_general`, `cayley_S3`,
  `cayley_S3_general`. See
  [Shared-basis automata](#shared-basis-automata-beyond-the-thesis).
- **Multi-start Mode I — `multi_start`** (**new, beyond the thesis**): Mode I
  with several start vertices. The chain is $m^2$ copies of one shared-basis
  automaton, one per guess of the total charge of the input (and Fourier index);
  they differ only by phases on the edges, every block is a Mode I chain, and
  the start weights keep the right guess. Unitary for every $N$ and all the data
  (one-line proof, in the docstring). It contains the counter ($m=1$) and overlaps
  Mode II (no edge phases), and reaches operators that no single start can
  represent at fixed bond: the parity-controlled phase has bond 4, a single start
  needs $j+1$ vertices at level $j$. Four named families: `multi_start_parity`,
  `multi_start_ring`, `multi_start_general`, `multi_start_stacked`. See section 4.3 of
  the [user guide](docs/userguide.md).
- **Single-tensor mechanisms** (`UniformMPUGenerator` only): `clock`,
  `group_automaton`, `shared_basis_automaton`, `cayley_automaton`,
  `self_direction`, `resonant`, `multi_symbol`, `invariant_subspace`,
  `degenerate_rows`. Not part of the thesis or paper and validated by the
  unit tests: `clock`, `group_automaton`, `shared_basis_automaton`,
  `cayley_automaton` and `degenerate_rows` are special cases of Mode I;
  `self_direction`, `resonant` and `multi_symbol` give on-site product
  unitaries.
- **Mixing combinators** (`mixing.py`): `stack` (operator composition),
  `nest` (rank-$k$ invariant subspace), `capped_sum` (a capped sum of two
  unitary chains, unitary by (RR) of its readout row, thesis
  `cor:mode1-readout`).
- **Two-layer checking** (with `verify=True`): for Modes I, II and the
  grafted composition the defining algebraic condition (the certificate) is
  checked, with the boundary-condition exceptions listed under
  [Verification](#verification-and-certification); every assembled chain also
  passes numerical safeguards (normalization for any $N$, plus a dense
  unitarity check to tolerance when $d^N \le 2^{10}$).

## Background

"Certified" here means something specific: for Modes I and II and the
grafted composition, unitarity follows from an *algebraic condition on
the site tensor itself*, proved once in the thesis; with `verify=True` the
condition is re-checked numerically on every draw (for Mode II, the boundary
condition only for the `cyclic` and `group` skeletons) — not inferred from
spot-checking a few chain lengths. Mode II-B (Lee–Yang) is the exception: no
algebraic certificate is checked at generation, and unitarity is validated
numerically only (normalization for any $N$, dense check to tolerance for
$d^N \le 2^{10}$).

- **Row Resolution (RR)**, Mode I: each row of the branching tensor must
  satisfy completeness ($\sum_u A_u^\dagger A_u = \mathbb{1}$) and
  cross-orthogonality between branches. The construction draws two Haar
  unitaries and builds each block from orthogonal projectors (any rank
  partition), which makes (RR) hold identically for any Haar draw —
  there is nothing to reject.
- **Fusion Closure (FC)**, Mode II: the channel matrices $y_u$ of a
  bond-diagonal tensor must close under
  $y_u^\dagger y_v = \sum_w \gamma^w_{uv} y_w$, and the boundary vector
  must solve one global quadratic equation. Each of the four skeletons
  supplies its own closed-form or Fourier-based solution to both
  conditions (e.g. `group`'s boundary uses a Peter–Weyl generalization of
  the flat-spectrum trick; it is built in closed form, not by a generic
  linear solve).
- **Intertwiner condition (I)**, Mode II-B: the product of two channel
  MPOs is related to the direct sum of the target channels by an
  injective (generally rectangular) intertwiner $V$ (Lee–Yang: $V$ is
  $9\times5$). `skeletons.solve_intertwiner` and
  `skeletons.verify_intertwiner` are standalone helpers for this condition;
  the generators do not call them, and `lee_yang_skeleton` builds its core
  tensors explicitly. The core tensors are fixed once per skeleton; only the
  per-site Haar dressing varies between draws.
- **Self-direction closed form**, translation-invariant-only
  constructions: the condition $\sum_b A[a,b]\,r_b = \mu\, r_a\, W$,
  solved by a tensor whose "junk" component on $r^\perp$ vanishes
  identically (these give on-site product unitaries; not part of the thesis
  or paper).

Beyond the per-mechanism certificate, `verify=True` runs two independent
numerical safeguards on every assembled chain: the normalization
$\mathrm{Tr}(U^\dagger U)/d^N = 1$ (one sweep, any $N$; necessary, not
sufficient) and, when $d^N \le 2^{10}$, the dense condition
$\|U^\dagger U - \mathbb 1\|_F \le 10^{-8}\sqrt{d^N}$. Either failure
raises `RuntimeError`.

## Install

TreeMPU is not on PyPI, so install it first:

```bash
pip install "TreeMPU @ git+https://github.com/Mrcocvs02/TreeMPU.git"
pip install git+https://github.com/Mrcocvs02/MPU_Generation.git
```

This package imports from [`TreeMPU`](https://github.com/Mrcocvs02/TreeMPU)
(blocking, boundary sweeps, dense reference), never the reverse.

For local development inside the parent monorepo (where this directory
is a git submodule):

```bash
git submodule update --init --recursive
pip install -e ./mpu_generators
```

## Quickstart

Mode I, general (non-uniform), with automatic blocking:

```python
from mpu_generators import MPUGenerator

gen = MPUGenerator(seed=0)
tensors, l, r = gen.branching(d=3, bond_dim=2, n_blocks=2, blocking="auto")
# tensors: list[np.ndarray], each A[p_in, p_out, bond_L, bond_R]
# l, r: boundary vectors; U_N = sum_{a,b} l[a] * <chain> * r[b]
```

Mode II, translationally invariant, non-abelian group skeleton:

```python
from mpu_generators import UniformMPUGenerator

gen = UniformMPUGenerator(seed=1)
tensors, l, r = gen.fusion(
    d=6,
    bond_dim=6,
    n_blocks=3,
    blocking=1,
    skeleton="group",
    group="S3",
)
# tensors[0] is identical to every other tensors[i] (TI construction)
assert all((tensors[0] == A).all() for A in tensors[1:])
```

Both snippets produce a `(tensors, l_vec, r_vec)` triple ready to hand to
[`TreeMPU.UniformMPU`/`GeneralMPU`](https://github.com/Mrcocvs02/TreeMPU/blob/main/README.md#usage-guide) for circuit
synthesis.

## API overview

Top-level exports:

```python
from mpu_generators import (
    MPUGenerator,
    UniformMPUGenerator,
    generate_random_mpu,  # generators
    build_family,
    FAMILIES,  # named families (thesis, paper, shared-basis automata)
    skeletons,
    mixing,  # Mode II-B skeletons, combinators
)
```

Every generation method — on either class — returns
`(tensors, l_vec, r_vec)`: `tensors` a `list[np.ndarray]` (one tensor per
block of `blocking` sites, i.e. one per site when `blocking=1`), plus the
left/right boundary vectors closing the chain.

**Conventions.**

- Tensors are `A[p_in, p_out, bond_L, bond_R]`: the first physical index is
  the input.
- The block stored at `A[:, :, v, u]` is the *transpose* of the operator
  $A[v,u]$ (rows = output) used in the thesis.
- `l_vec` and `r_vec` contract linearly, *without* complex conjugation:
  $U_N=\sum \ell_{v_0} r_{v_N}\,A[v_0,v_1]\otimes\cdots\otimes A[v_{N-1},v_N]$.
- Sites are built in construction order and reversed at the end
  (`_finalize_chain`). With per-site resampling, chains for increasing $N$
  therefore extend a common sequence at the *left* end (Mode II-B, which is
  not reversed: at the right end).
- Dense matrices from `TreeMPU.checks.dense_unitary_from_tensors` are
  `U[out, in]` with site 0 the least significant digit (Qiskit order).

### One pipeline, one skeleton per mode

Every construction is a *skeleton* (`mpu_generators.skeletons`): the data that
define it, with three hooks — `prepare` (chain-level draws: a digraph, a
boundary vector), `site` (one tensor) and `close` (boundary vectors, a readout
cascade). The generator owns the rest (`MPUGenerator.sample(skeleton, ...)`:
size checks, per-site or uniform sampling, blocking, verification), so all
modes run through the same code and the methods below are thin wrappers:

| method | skeleton |
|---|---|
| `branching` | `RowResolutionSkeleton` |
| `fusion(skeleton="cyclic" / "nested_flag" / "matrix_units" / "group")` | `CyclicSkeleton`, `NestedFlagSkeleton`, `MatrixUnitSkeleton`, `GroupSkeleton` (base class `FusionClosureSkeleton`) |
| `fusion_bond_resolved` | `BondResolvedSkeleton` (alias `FusionSkeleton`; e.g. `lee_yang_skeleton()`) |
| `grafted` | `GraftedSkeleton` |
| `shared_basis_chain`, `cayley_chain` | `SharedBasisSkeleton` |
| `multi_start` | `MultiStartSkeleton` |

A new construction is one more subclass of `Skeleton` (or of
`FusionClosureSkeleton` for a new fusion rule) passed to `sample`; nothing in
the generator changes. The outputs of all existing methods are unchanged bit
for bit (`tests/test_golden_outputs.py`). One legacy difference is kept for
reproducibility of the thesis instances: Mode II-B draws its sites in physical
order (`reverse_sites = False`).

### `MPUGenerator(seed=0, verify=True, coherent_in_n=True)`

With `coherent_in_n=True` (default) the generator is seeded with `seed`
alone: one random family for every chain length (the uniform site tensor
does not depend on `n_sites`, and resampled chains of different lengths
share their draws; the tensor-first sampling paper relies on this). With
`coherent_in_n=False` every chain length gets an independent draw, seeded
with `(seed, n_sites)`.

Every mode below takes **exactly one** of `n_sites` (raw physical chain
length) or `n_blocks` (number of final coarse-grained tensors), plus a
`blocking` factor (an `int`, or `"auto"` to search for the smallest
factor satisfying Assumption 1, capped by `max_blocking`).

| Method | Mode | Key parameters / constraints |
|---|---|---|
| `branching(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8, readout=None, digraph=None, out_degree=None, ranks="balanced")` | I | `readout`: list of vertices for combined multi-vertex readout, any auxiliary dimension; it appends $m=\max(1,\lceil\log_d\lvert\text{readout}\rvert\rceil)$ sites, so `blocking` must divide both `n_sites` and `n_sites + m` (hence $m$); `digraph`: hand-picked `{v: [targets]}` (1 to `d` distinct targets per vertex, used at every site); `out_degree`: int or `"random"` (uniform in `1..min(d, bond_dim)` per vertex); `ranks`: `"balanced"`, `"random"` (uniform composition of `d`) or hand-picked `{v: [ranks]}` (needs `digraph`). Non-default choices are different ensembles on the same certified class. |
| `fusion(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8, skeleton="cyclic", group="Zn", group_n=3)` | II | `skeleton in {"cyclic","nested_flag","matrix_units","group"}`; `nested_flag` needs `bond_dim <= d`; `cyclic` accepts any `bond_dim` (a group convolution only when `bond_dim` divides `d`; for `bond_dim > d` the single-site channels are linearly dependent and Assumption 1 first holds on blocks of `ceil((bond_dim-1)/(d-1))` sites, so use `blocking="auto"`); `matrix_units` needs `bond_dim = 1 + m**2` for integer `m <= d` (`bond_dim` = 2, 5, 10, 17, 26, ...); `group` needs `d = bond_dim = |G|` with `group in {"Zn","Zn_product","S3"}` (for `Zn`, `|G| = group_n`) |
| `fusion_bond_resolved(skeleton, n_sites=None, n_blocks=None, blocking=1, max_blocking=8)` | II-B | `skeleton`: a pre-built `FusionSkeleton` (e.g. from `mpu_generators.skeletons.lee_yang_skeleton()`) |
| `grafted(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8)` | I + II | requires even `bond_dim` (`= 2 * n_channels`) with `n_channels <= d`; the channels are the `cyclic` skeleton, a group convolution only when `n_channels` divides `d` |
| `multi_start(n_sites=None, n_blocks=None, d=2, n_sectors=2, D=4, transitions=None, R=None, V=None, final_phases=None, start=0, charges=None, sector_phases="random", edge_phases="random", blocking=1, max_blocking=8)` | I, several starts | automaton data as in `shared_basis_chain`; bond $m^2D$ with $m$ = `n_sectors`, $mD$ with `edge_phases=None`; use an odd ring or maps that do not track the charge, otherwise Assumption 1 fails |
| `shared_basis_chain(n_sites=None, n_blocks=None, d=2, D=4, transitions=None, R=None, V=None, final_phases=None, start=0, blocking=1, max_blocking=8)` | I | one reading basis per site, `D` an int or a list of `n+1` bond dimensions (site-dependent class only); `transitions`: `"permutation"` (default for constant `D`), `"ring"`, `"random"` or explicit `[a][v]` / `[k][a][v]`; `R`, `V`: one for all sites or one per site; `final_phases` an array or `"random"` |
| `cayley_chain(n_sites=None, n_blocks=None, group="S3", generators=None, group_n=0, R=None, V=None, final_phases=None, blocking=1, max_blocking=8)` | I | vertices are the group elements, letters the `generators` (default for `S3`: an involution and a 3-cycle, so `d=2`, `D=6`); same group at every site, site-dependent `R` and `V` |

`blocking="auto"` retries increasing block sizes until the boundary
operators have full rank at every cut (minimal bond dimension, as checked
by `TreeMPU.utils.satisfies_assumption_1`; required by
`TreeMPU.GeneralMPU(canonicalize=False)`), capped by `max_blocking`. Every
block size, automatic or explicit, is refused before allocation if the
blocked tensor would exceed 512 MiB (the physical dimension grows as
`d**blocking`).

### `UniformMPUGenerator(MPUGenerator)`

```python
class UniformMPUGenerator(MPUGenerator):
    _resample = False
```

That single class-attribute override is the entire translational-invariance
mechanism: the base class draws a fresh site dressing per call when
`_resample=True` (the default), or draws once and reuses `[site] * n_sites`
when `False`. This alone turns `branching`/`fusion`/`fusion_bond_resolved`/
`grafted` into their translationally-invariant counterparts with no
duplicated algebra. On top of that, `UniformMPUGenerator` adds
translationally-invariant-*only* mechanisms, each returning a single
reused bulk tensor `(A, l_vec, r_vec)` (use `[A] * N`):

| Method | Recipe |
|---|---|
| `clock(D=3, d=2, seed=None)` | A $D$-cycle permutation dressed by $D$ independent Haar unitaries — the degree-1 special case of (RR) |
| `group_automaton(group="Zn", n=3, cocycle=None, seed=None)` | Group-multiplication automaton $A[g,h]=\lvert h\rangle\langle g^{-1}h\rvert$, optionally dressed by a 2-cocycle; `group in {"Zn","Sn"}` |
| `shared_basis_automaton(transitions=None, D=4, d=2, R=None, V=None, final_phases=None, start=0, seed=None)` | Counter automaton $A[v,\sigma_a(v)]\mathrel{+}=V_v\lvert r_a\rangle\langle r_a\rvert$: every vertex reads the site in the same basis $R$; letter $a$ moves the vertex by $\sigma_a$ (default: the lazy ring $v\mapsto v+a$), $V_v$ Haar by default, $r_u=e^{i\varphi_u}$. Unitary for every $R$, $V$, $\varphi$ and every map $\sigma_a$ |
| `cayley_automaton(group="S3", generators=None, group_n=0, R=None, V=None, final_phases=None, seed=None)` | `shared_basis_automaton` on the Cayley graph of `group` (vertex $\to$ vertex $\cdot g_a$, $d$ = number of generators; default for `S3`: a transposition and a 3-cycle, $d=2$, $D=6$) |
| `self_direction(D=4, d=2, mu=1.0, W=None, seed=None)` | A tensor satisfying $\sum_b A[a,b]\,r_b = \mu\, r_a\, W$ |
| `resonant(D=5, d=2, target_phases=None, seed=None)` | Spectrum on the $D$-th roots of unity, embedded via a non-unitary change of basis that is still diagonalizable with unit-modulus eigenvalues |
| `multi_symbol(d=3, seed=None)` | Special case of `self_direction` with $\mu=1$, $W=\operatorname{diag}(\mu_0,\dots,\mu_{d-1})$ — not an independent mechanism |
| `invariant_subspace(sub_A, sub_l, sub_r, D_extra=2, seed=None)` | Any other TI mechanism's output restricted, in a generic basis, to a closed invariant subspace containing $r$ |
| `degenerate_rows(D=2, d=2, groups=None, seed=None)` | Literal row duplication of a Mode I tensor — (RR) still holds per row regardless of duplication |

Two variants are not implemented: dressing a unitary on ancilla ⊗ physical
does not produce (RR) (the resolutions of the identity it gives sum over the
wrong index), and a generic triangular (Jordan-type) extension leaves a
boundary mismatch with the bulk. The anticommuting triangular tensor does work,
and is a Mode I chain in a site-dependent gauge: see `docs/userguide.md`
section 4.3, which also covers chains with several start vertices.

#### Shared-basis automata (beyond the thesis)

"Shared" refers to the vertices: at a given site all vertices read the input
in the same basis, so the input-side projectors take only $d$ distinct values
instead of one per edge. The basis does not have to be the same at different
sites (see the next subsection). `branching`, in contrast, draws an
independent Haar basis at every vertex. The class
is exactly unitary either way; what changes is how an operator spreads. The
table gives the fraction of $U_N P_i U_N^\dagger$ that is *not* supported in a
window of radius $R$ around the middle site ($N=60$, maximum over the three
Pauli matrices, three seeds; computed by contracting the operator as a
bond-dimension-$D^2$ MPO, checked against dense matrices at $N\le 10$):

| tensor ($d=2$) | $R=4$ | $R=12$ | $R=24$ |
|---|---|---|---|
| `branching`, ring digraph $D=4$ (Haar bases) | 0.67-0.70 | 0.26-0.30 | 0.05-0.09 |
| `branching`, ring digraph $D=6$ (Haar bases) | 0.78-0.80 | 0.54-0.56 | 0.28-0.31 |
| `branching`, random digraph $D=4$ | 0.33-0.75 | 0.05-0.36 | 0.002-0.13 |
| `shared_basis_automaton`, ring $D=4$ | 0.89-0.91 | 0.84-0.87 | 0.83-0.86 |
| `shared_basis_automaton`, ring $D=6$ | 0.92-0.94 | 0.88-0.91 | 0.85-0.88 |
| `cayley_automaton("S3")`, $D=6$ | 0.90-0.97 | 0.84-0.94 | 0.82-0.91 |

With permutation maps the weight towards larger site index does not decay
(for the seed-0 tensors at $N=40$ it is $0.823$ for every $R=2,6,12$) while
the weight towards smaller site index decays (0.52, 0.29, 0.12 for the ring
$D=4$): the chain is a one-directional counter and is not reflection
symmetric. Mode II `cyclic` (identity $V$) also has an $R$-independent weight
but is symmetric. These are numerical observations, not theorems. Everything
else measured is the same as for the Haar-basis ring: the operator-Schmidt
rank at the middle cut ($N=10$) equals the bond dimension, and
the circuits of [`TreeMPU`](https://github.com/Mrcocvs02/TreeMPU) with the
closed-form Mode I boundary have identical $q=\chi$, $\ell$, qubit count,
depth and gate count (8 sites, $D=4$: $q=4$, $\ell=3$, 29 qubits), because
that cost depends only on $\chi$ and $d$.

#### Site-dependent chains: what makes the tail non-decaying

The unitarity proof only uses the first site at which two input strings
differ: the prefix is shared, so the vertex is shared, and the two inputs are
read in orthogonal vectors of one site. The reading basis therefore only has
to be common to the vertices *of that site*; it can change from site to site,
as can the vertex unitaries, the maps and the bond dimensions
(`MPUGenerator.shared_basis_chain`). The left boundary must be a single basis
vector (several start vertices could make paths collide), and the right
boundary must be unimodular.

Fraction of $U_N P_0 U_N^\dagger$ outside a window of radius $R$ around site 0
($N=8$, $d=2$, $D=4$, bases and unitaries independent at every site, maximum
over $X$ and $Z$, six seeds, dense computation; mean and range):

| maps | $R=1$ | $R=2$ | $R=3$ |
|---|---|---|---|
| ring | 0.83 (0.63-0.99) | 0.83 (0.63-0.99) | 0.83 (0.63-0.99) |
| random permutations | 0.81 (0.55-0.96) | 0.81 (0.55-0.95) | 0.81 (0.55-0.95) |
| random maps | 0.41 (0.00-0.90) | 0.32 (0.00-0.71) | 0.25 (0.00-0.59) |

With permutation maps the weight does not decrease with the radius (all of
it lies beyond $R=3$; per seed the weight changes by at most 0.005 between $R=1$ and $R=3$) even though the bases differ from site to site; with
coalescing maps it decreases. The reason is not that the basis is the same
at every site (the ring gives 0.88 with one basis, translation invariant, and
0.83 with independent bases at every site): the vertex $v_{k-1}$ is a deterministic function of the letters on
the left, the output at site $k$ depends on it, and with permutation maps no
information about the letters is lost, so a changed letter changes every
later output. In `branching` the vertices read in different bases, the path
is a superposition rather than a function of fixed letters, and the tail
decays (numerical; this explanation is an interpretation, not a theorem).

*Interpolating between the two cases.* Let the vertices of the ring read in
$g$ groups, each group with its own Haar basis ($g=1$ is the shared-basis
ring, $g=D$ one basis per vertex, as in `branching` with the same digraph; all
are Mode I tensors, so exactly unitary). Weight of $U_8X_0U_8^\dagger$ outside
a window of radius $R$ ($d=2$, $D=4$, mean of 8 seeds):

| groups | $R=1$ | $R=2$ | $R=3$ |
|---|---|---|---|
| $g=1$ | 0.98 | 0.98 | 0.98 |
| $g=2$ | 0.68 | 0.60 | 0.55 |
| $g=4$ (one basis per vertex) | 0.70 | 0.59 | 0.51 |
| library `branching` ring | 0.58 | 0.50 | 0.44 |

Changing the *number* of Haar bases is not a knob: a second Haar basis already
gives the decaying behaviour of the Haar-basis ring (the same at $D=6$ for
$g=2,3,6$, without a monotone trend in $g$). Changing the *angle* is. Let the
vertices read in $R_v=R_0\,e^{i\theta\,\vec n_v\cdot\vec\sigma}$, with $\vec n_v$ a
random axis per vertex ($\theta=0$ is the shared basis). Drop of the weight
between $R=1$ and $R=3$ ($d=2$, ring, $N=8$, mean of 8 seeds; a flat tail is
a drop of about 0):

| $\theta$ | 0 | 0.1 | 0.2 | 0.4 | 0.8 | $\pi/2$ |
|---|---|---|---|---|---|---|
| $D=4$ | 0.01 | 0.02 | 0.05 | 0.12 | 0.16 | 0.16 |
| $D=6$ | 0.01 | 0.02 | 0.04 | 0.11 | 0.17 | 0.11 |

The tail stays flat for small angles and decays progressively, saturating near
the value of the Haar-basis ring (0.16 at $D=4$) for $\theta\gtrsim0.8$. Other
targeted choices at $D=4$ (drop between $R=1$ and $R=3$): two mutually unbiased
bases alternating around the ring, 0.16 (decays like Haar); the *same* basis
with the letter labels swapped on odd vertices, 0.00 (stays flat: it is still a
shared-basis automaton, with vertex-dependent maps); a single vertex in another
Haar basis, 0.09 at $D=4$ and 0.04 at $D=6$ (partial). So the decay appears
when a path passes through vertices that read in a different set of
projectors, and its strength grows with how different, and how many, those
vertices are. This is a finite-size observation (window $R\le3$, $N=8$, 8
seeds; the saturation value fluctuates by a few hundredths), not a theorem. The zeros are draws in which
$\sigma_{0,0}$ and $\sigma_{0,1}$ send the start vertex to the same vertex,
so that the letter at site 0 never reaches the later vertices (tested: the
weight is exactly zero then, and above 0.5 otherwise in the test). These are finite-size
numerical observations, not theorems. `TreeMPU.GeneralMPU` synthesizes a
site-dependent ring exactly with its canonical boundaries (seed 0, $d=2$,
$D=3$, two blocks of two sites: $q=3.9$, error $3\times10^{-14}$); random permutations or maps may fail Assumption 1 for
every block size, because the start vertex then reaches fewer than $D$
vertices.

`group_automaton` is the special case with reading basis $1$, all group
elements as letters and $V_g$ the left multiplication (identical tensors,
tested for $\mathbb Z_2,\mathbb Z_3,\mathbb Z_4,S_3$); Mode II `cyclic` with
`bond_dim == d` is the special case $V_v=1$ with phases
$\varphi_q=\arg\sum_g\ell_g\omega^{gq}$ in the eigenbasis of $y_1$ (tested).

### `generate_random_mpu(...)`

```python
generate_random_mpu(
    n_sites=None,
    d=2,
    bond_dim=2,
    mode="branching",
    blocking=1,
    uniform=False,
    seed=0,
    verify=True,
    n_blocks=None,
    max_blocking=8,
    **mode_kwargs,
)
```

A back-compat/convenience dispatcher over the class methods above
(`mode in {"branching","mode1","fusion","mode2","grafted"}`), forwarding
extra keyword arguments (e.g. `skeleton=` for fusion, `readout=` for
branching) via `**mode_kwargs`. `fusion_bond_resolved` (Mode II-B) is
**not** reachable through this dispatcher — it needs a pre-built
`FusionSkeleton` object rather than scalar keyword arguments, so call
`MPUGenerator.fusion_bond_resolved`/`UniformMPUGenerator.fusion_bond_resolved`
directly for it. The dispatcher has no `coherent_in_n` argument (the
generator keeps its default `True`).

## Mixing combinators

`mpu_generators.mixing` combines two unitary `(tensors, l, r)` triples (for
example, outputs of the generators above) into a new unitary one:

| Function | Construction | Requirements |
|---|---|---|
| `stack(mpu1, mpu2, seed=None)` | Operator composition $U_N := U_N^{(1)} \cdot U_N^{(2)}$, bond dimension $D_1 D_2$, no extra site — unitary because both factors are; no further check | equal $d$, equal chain length |
| `nest(sub_A, sub_l, sub_r, D_extra=2, seed=None, generator=None)` | Thin alias for `UniformMPUGenerator.invariant_subspace`; with `seed=None` the embedding uses the generator's own seed (0 for a newly created generator) | — |
| `capped_sum(mpu1, mpu2, seed=None)` | One extra site prepended, carrying a row-resolution pair $\Pi_1,\Pi_2$ built from the same recipe Mode I uses internally: $U_{N+1}=\Pi_1\otimes U_N^{(1)}+\Pi_2\otimes U_N^{(2)}$, unitary for *any* two unitary inputs by (RR) of the readout row (thesis `cor:mode1-readout`) | equal $d$, equal chain length; bond dimensions may differ |

Deliberately **not** implemented: naive summing without the extra capped
site ($l = l_1 + l_2$, no $\Pi$) — this fails unitarity in general (there
is no reason for the $U^{(1)\dagger}U^{(2)}$ cross terms to cancel); the test
suite uses it as a negative control.

## Verification and certification

With `verify=True` (the default) each mechanism's algebraic certificate is
checked immediately after construction, where a checker exists:

- Mode I — `_check_row_resolution` / `_check_combined_readout_cascade`:
  completeness and cross-orthogonality per row/vertex.
- Mode II — `_check_fusion_closure` (cyclic), `_check_nested_flag_closure`,
  `_check_matrix_unit_closure`, `_check_fusion_closure_general`
  (group), plus `_check_channel_span_full_rank`/
  `_check_group_representation` for the channel/representation structure
  itself. The boundary condition (B) is checked only for `cyclic`
  (`_check_flat_spectrum`) and `group` (`_check_group_boundary`); the
  `nested_flag` and `matrix_units` boundaries are built in closed form and
  not checked.
- Grafted — (RR) of the Mode I part, closure of the cyclic channels and the
  flat-spectrum boundary.
- Mode II-B — no algebraic certificate is checked. The helpers
  `skeletons.solve_intertwiner` and `skeletons.verify_intertwiner` (residual
  of the intertwiner condition for a given $V$) are standalone and are not
  called at generation; the null-complement and boundary-compatibility
  conditions of the thesis are not implemented.

On top of that, every assembled chain goes through the numerical
safeguards described in [Background](#background) (normalization, and the
dense check to tolerance for $d^N\le2^{10}$), regardless of which
per-mechanism certificate, if any, was used to build the tensor. For Mode II-B
these safeguards are the only unitarity check. The test suite
(`tests/test_mpu_generators.py`) mirrors this same two-layer structure and
additionally builds fully dense $d^N \times d^N$ matrices for direct
$U^\dagger U = \mathbb{1}$ checks at small $N$, as an independent
cross-check of the whole pipeline, not just of the certificate functions in
isolation.

**Cost.** The checks are for implementation errors and for user-supplied
input (hand-picked digraphs, bases, maps, skeleton tensors), not for the
mathematics, which is analytic. Measured CPU time (one thread) of
`verify=True` beyond the generation itself: about 60 ms at tiny $N$ (the dense
check, $d^N\le2^{10}$), 0.03 s at $N=64$, $D=4$; 0.6 s at $N=4096$, $D=4$
(generation 2.2 s); about 1 s at $D=32$–$64$ ($N=64$–$128$, generation 0.5 s). For
large sweeps pass `verify=False` (also in `generate_random_mpu`); the output is
identical, only the checks are skipped.

## Example: Mode II-B / Lee-Yang

```python
from mpu_generators import MPUGenerator
from mpu_generators.skeletons import lee_yang_skeleton

sk = lee_yang_skeleton()
gen = MPUGenerator(seed=9, verify=True)
tensors, l, r = gen.fusion_bond_resolved(sk, n_blocks=2, blocking=1)
# d = 5, bond dimension 14 (Lee-Yang/Fibonacci channels {1, O_e, O_sigma})
```

## Relationship to TreeMPU

This package imports from
[`TreeMPU`](https://github.com/Mrcocvs02/TreeMPU):

- `TreeMPU.utils.block_mpu_tensors` — coarse-grains consecutive site
  tensors for the `blocking`/`blocking="auto"` parameters above.
- `TreeMPU.utils.compute_chain_boundary_operators` — the normalization
  safeguard, and `TreeMPU.utils.satisfies_assumption_1` — the full-rank
  boundary check driving `blocking="auto"`.
- `TreeMPU.checks.dense_unitary_from_tensors` — the dense reference matrix.

The dependency direction is one-way by design: this package imports from
`TreeMPU`, never the reverse, so it can be split into its own
independently-versioned package with no rework — which is exactly its
current status as a separate repository, included here as a git
submodule.

## Testing

```bash
cd mpu_generators
pytest
```

(or `pytest` from the parent repo's root, which already points
`pythonpath` at both packages.) `tests/test_mpu_generators.py` covers
every mechanism above — Mode I (incl. combined readout, general and
uniform), Mode II (all four skeletons, incl. bad-shape/dimension
`ValueError` cases and a dedicated test that the representation-homomorphism
checker catches a broken representation), Mode II-B (a synthetic
intertwiner round-trip plus the Lee-Yang skeleton compared with a
separately typed copy of its tables, which guards against accidental
edits), `grafted`, every
translationally-invariant-only mechanism (including two documented
negative/edge-case results: `group_automaton`'s combined readout is *not*
free, and `resonant`'s transfer matrix can be non-unitary while the chain
itself stays exactly unitary), all three mixing combinators, and the
explicit negative control `test_naive_sum_without_capped_site_fails`.
`tests/test_certificates_and_api.py` checks that every algebraic
certificate rejects a perturbed input, the rectangular Lee-Yang
intertwiner, the Lee-Yang boundary family, the family registry and
reproducibility.

## Reference

The theory implemented here is derived in the author's master's thesis,
*Beyond-QCA MPUs: Generation, Certification, and Circuit Synthesis*
(M. Vassallo, University of Bologna, 2026;
[AMS Laurea](https://amslaurea.unibo.it/id/eprint/39893), with
known errors listed in
[docs/thesis_errata.md](https://github.com/Mrcocvs02/TreeMPU/blob/main/docs/thesis_errata.md)),
and in the manuscript "Tensor-First Sampling of Exact
Matrix-Product Unitaries" (in preparation). It builds on:

> G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum Circuits for
> Matrix-Product Unitaries", *Phys. Rev. Lett.* **135**, 260602 (2025).
> [arXiv:2508.08160](https://arxiv.org/abs/2508.08160),
> [doi:10.1103/yshb-hmml](https://doi.org/10.1103/yshb-hmml).

```bibtex
@article{Styliaris2025,
  title   = {Quantum Circuits for Matrix-Product Unitaries},
  author  = {Styliaris, Georgios and Trivedi, Rahul and Cirac, J. Ignacio},
  journal = {Physical Review Letters},
  volume  = {135},
  pages   = {260602},
  year    = {2025},
  doi     = {10.1103/yshb-hmml},
  eprint  = {2508.08160},
  archivePrefix = {arXiv}
}
```

## Status

Alpha (0.3.0). Used as a submodule of
[`TreeMPU`](https://github.com/Mrcocvs02/TreeMPU). All
mechanisms listed above are implemented and covered by the test suite.

## Roadmap

Not implemented yet (contributions welcome):

- **Group convolution for arbitrary finite groups** from user-supplied
  multiplication tables and irreducible representations (`Zn`, `Zn_product`
  and `S3` are implemented).
- **A generic Mode II boundary sampler** by lifting a unitary element of the
  fused-channel algebra (exponential of a Hermitian element) through the
  injective map $\Gamma_{N_0}$; requires checking injectivity at some $N_0$.
- **New skeletons**: $\mathrm{Fun}(G)$ (diagonal, unit-modulus entries),
  Brandt, $H_8$, and a generator for periodic boundaries with a prescribed
  GNVW index.
- **Mode II-B**: an algebraic certificate (null-complement and boundary
  compatibility conditions of the thesis) beyond the intertwiner (I).
- **Analysis layer**: Mode II causal multipliers and the Mode I endpoint
  transfer, currently only in the thesis production scripts.

## AI usage disclosure

Generative AI (Claude, Anthropic) was used during the development of this
package for guided code generation, testing, analysis and formatting. All
development was directed by the author, who is responsible for the contents
of this repository; the behaviour of the code is validated by the test suite
and by the numerical verifications described in the documentation.

## License

MIT — see [LICENSE](LICENSE).
