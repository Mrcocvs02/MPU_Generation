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
- **Translationally-invariant-only mechanisms** (`UniformMPUGenerator`):
  `clock`, `group_automaton`, `shared_basis_automaton`, `cayley_automaton`,
  `self_direction`, `resonant`, `multi_symbol`, `invariant_subspace`,
  `degenerate_rows` — single-tensor constructions that are not part of the
  thesis or paper and are validated by the unit tests only: `clock`,
  `group_automaton`, `shared_basis_automaton`, `cayley_automaton` and
  `degenerate_rows` are special cases of Mode I; `self_direction`,
  `resonant` and `multi_symbol` give on-site product unitaries.
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
    FAMILIES,  # named thesis/paper families
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
wrong index), and a triangular (Jordan-type) extension leaves a boundary
mismatch with the bulk.

#### Shared-basis automata compared with the Haar-basis Mode I sampler

`branching` draws an independent Haar basis at every vertex. In
`shared_basis_automaton` all vertices read in one basis, so the input-side
projectors take only $d$ distinct values instead of one per edge. The class
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
