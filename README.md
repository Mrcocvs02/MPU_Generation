# mpu_generators

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2508.08160-b31b1b.svg)](https://arxiv.org/abs/2508.08160)

Certified random generators for Matrix Product Unitaries (MPUs): sample
tensors that are unitary for every chain length $N$ under a
single fixed generative rule, rather than approximately unitary or
unitary only on average. Every construction confines the relevant
transfer object to an $N$-independent algebraic family *by design*: an
identity of the local tensor and boundary data implies unitarity for every
length — nothing here relies on statistical typicality, large-$N$
concentration, or rejection sampling.

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
- [License](#license)

## Key features

- **Mode I — `branching`** (row resolution): a random digraph on the
  bond dimension whose rows are filled by Haar-dressed, rank-balanced
  orthogonal blocks; completeness holds for *any* Haar draw, no rejection
  needed. Supports combined multi-vertex readout via an auxiliary
  cascade of sites.
- **Mode II — `fusion`** (fusion closure): a bond-diagonal tensor whose
  channel matrices close under a fixed fusion rule. Four built-in
  skeletons — `cyclic` ($\mathbb{Z}_M$ characters), `nested_flag`
  (shrinking-flag projectors, the diagonal-phase/MCZ-type family),
  `matrix_units` (identity channel + matrix units), and `group` (regular
  representation of any finite group, abelian or not — `Zn`,
  `Zn_product`, `S3`).
- **Mode II-B — `fusion_bond_resolved`** (bond-resolved fusion): per-channel
  core tensors fixed once by solving the Intertwiner condition, then only
  the per-site Haar dressing is resampled. Ships with the Lee-Yang/Fibonacci
  skeleton (`lee_yang_skeleton`, $d=5$, bond dimension 14) out of the box.
- **Grafted composition** (thesis `thm:grafted-factorization`): a Mode I two-vertex path algebra
  grafted inside the channels of a Mode II cyclic-convolution skeleton —
  a product of two independently certified unitaries.
- **Translationally-invariant-only mechanisms** (`UniformMPUGenerator`):
  `clock`, `group_automaton`, `self_direction`, `resonant`,
  `multi_symbol`, `invariant_subspace`, `degenerate_rows` — single-tensor
  constructions from working notes, not part of the thesis or paper:
  `clock`, `group_automaton` and `degenerate_rows` are special cases of
  Mode I; `self_direction`, `resonant` and `multi_symbol` give on-site
  product unitaries.
- **Mixing combinators** (`mixing.py`): `stack` (operator composition),
  `nest` (rank-$k$ invariant subspace), `capped_sum` (a capped, provably
  unitary sum of two independently certified chains).
- **Two-layer checking** on every construction: the defining algebraic
  condition (the certificate) is checked directly, and the assembled chain
  passes numerical safeguards (normalization for any $N$, plus an exact
  dense unitarity check when $d^N \le 2^{10}$).

## Background

"Certified" here means something specific: for Modes I and II and the
grafted composition, unitarity is guaranteed by an *algebraic condition on
the site tensor itself*, proved once and checked directly on every draw —
not inferred from spot-checking a few chain lengths. Mode II-B (Lee–Yang)
is the exception: only the intertwiner condition is checked, and global
unitarity is verified numerically (dense for $d^N \le 2^{10}$,
normalization beyond).

- **Row Resolution (RR)**, Mode I: each row of the branching tensor must
  satisfy completeness ($\sum_u A_u^\dagger A_u = \mathbb{1}$) and
  cross-orthogonality between branches. The construction draws two Haar
  unitaries and builds each block from rank-balanced orthogonal
  projectors, which makes (RR) hold identically for any Haar draw —
  there is nothing to reject.
- **Fusion Closure (FC)**, Mode II: the channel matrices $y_u$ of a
  bond-diagonal tensor must close under
  $y_u^\dagger y_v = \sum_w \gamma^w_{uv} y_w$, and the boundary vector
  must solve one global quadratic equation. Each of the four skeletons
  supplies its own closed-form or Fourier-based solution to both
  conditions (e.g. `group`'s boundary uses a Peter–Weyl generalization of
  the flat-spectrum trick — the generic linear solver was checked *not*
  to work there).
- **Intertwiner condition (I)**, Mode II-B: the product of two channel
  MPOs must be related to the direct sum of the target channels by an
  injective (generally rectangular) intertwiner $V$
  (`skeletons.solve_intertwiner`; Lee–Yang: $V$ is $9\times5$). The core
  tensors are fixed once per skeleton; only the per-site Haar dressing
  varies between draws.
- **Self-direction closed form**, translation-invariant-only
  constructions: an ansatz $\sum_b A[a,b]\,r_b = \mu\, r_a\, W$ whose "junk"
  component on $r^\perp$ vanishes identically (these give on-site product
  unitaries; not part of the thesis or paper).

Beyond the per-mechanism certificate, `verify=True` runs two independent
numerical safeguards on every assembled chain: the normalization
$\mathrm{Tr}(U^\dagger U)/d^N = 1$ (one sweep, any $N$; necessary, not
sufficient) and, when $d^N \le 2^{10}$, the exact dense condition
$U^\dagger U = \mathbb 1$. Either failure raises `RuntimeError`.

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
`(tensors, l_vec, r_vec)`: `tensors` a `list[np.ndarray]` (one bulk
tensor per site, convention `A[p_in, p_out, bond_L, bond_R]`), plus the
left/right boundary vectors closing the chain.

### `MPUGenerator(seed=0, verify=True)`

Every mode below takes **exactly one** of `n_sites` (raw physical chain
length) or `n_blocks` (number of final coarse-grained tensors), plus a
`blocking` factor (an `int`, or `"auto"` to search for the smallest
factor satisfying Assumption 1, capped by `max_blocking`).

| Method | Mode | Key parameters / constraints |
|---|---|---|
| `branching(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8, readout=None)` | I | `readout`: list of vertices for combined multi-vertex readout, any auxiliary dimension |
| `fusion(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8, skeleton="cyclic", group="Zn", group_n=3)` | II | `skeleton in {"cyclic","nested_flag","matrix_units","group"}`; `cyclic` needs `bond_dim <= d`; `matrix_units` needs `bond_dim = 1 + m**2` for integer `m` (2, 5, 10, 17, 26, ...); `group` needs `d = bond_dim = |G|`, `group in {"Zn","Zn_product","S3"}` |
| `fusion_bond_resolved(skeleton, n_sites=None, n_blocks=None, blocking=1, max_blocking=8)` | II-B | `skeleton`: a pre-built `FusionSkeleton` (e.g. from `mpu_generators.skeletons.lee_yang_skeleton()`) |
| `grafted(d=2, bond_dim=2, n_sites=None, n_blocks=None, blocking=1, max_blocking=8)` | I + II | requires even `bond_dim` (`= 2 * n_channels`) |

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
| `self_direction(D=4, d=2, mu=1.0, W=None, seed=None)` | The $(\star)$ ansatz $\sum_b A[a,b]\,r_b = \mu\, r_a\, W$ |
| `resonant(D=5, d=2, target_phases=None, seed=None)` | Spectrum on the $D$-th roots of unity, embedded via a non-unitary change of basis that is still diagonalizable with unit-modulus eigenvalues |
| `multi_symbol(d=3, seed=None)` | Special case of `self_direction` with $\mu=1$, $W=\operatorname{diag}(\mu_0,\dots,\mu_{d-1})$ — not an independent mechanism |
| `invariant_subspace(sub_A, sub_l, sub_r, D_extra=2, seed=None)` | Any other TI mechanism's output restricted, in a generic basis, to a closed invariant subspace containing $r$ |
| `degenerate_rows(D=2, d=2, groups=None, seed=None)` | Literal row duplication of a Mode I tensor — (RR) still holds per row regardless of duplication |

Two variants from the working notes are not implemented because they
fail: dressing a unitary on ancilla ⊗ physical does not produce (RR) (the
resolutions of the identity it gives sum over the wrong index), and a
triangular (Jordan-type) extension leaves a boundary mismatch with the
bulk.

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
directly for it.

## Mixing combinators

`mpu_generators.mixing` combines two already-certified `(tensors, l, r)`
triples into a new, still-certified one:

| Function | Construction | Requirements |
|---|---|---|
| `stack(mpu1, mpu2, seed=None)` | Operator composition $U_N := U_N^{(1)} \cdot U_N^{(2)}$, bond dimension $D_1 D_2$, no extra site — trivially unitary, no certification beyond the inputs' own | equal $d$, equal chain length |
| `nest(sub_A, sub_l, sub_r, D_extra=2, seed=None, generator=None)` | Thin alias for `UniformMPUGenerator.invariant_subspace` | — |
| `capped_sum(mpu1, mpu2, seed=None)` | One extra site prepended, carrying a row-resolution pair $\Pi_1,\Pi_2$ built from the same recipe Mode I uses internally: $U_{N+1}=\Pi_1\otimes U_N^{(1)}+\Pi_2\otimes U_N^{(2)}$, provably unitary for *any* two certified inputs | equal $d$, equal chain length; bond dimensions may differ |

Deliberately **not** implemented: naive summing without the extra capped
site ($l = l_1 + l_2$, no $\Pi$) — this fails unitarity in general (there
is no reason for the $U^{(1)\dagger}U^{(2)}$ cross terms to cancel), and a
regression test in `tests/` demonstrates the failure directly rather than
leaving it as an unimplemented footnote.

## Verification and certification

Each mechanism's algebraic certificate is checked immediately after
construction (gated by `verify=True`, the default):

- Mode I — `_check_row_resolution` / `_check_combined_readout_cascade`:
  completeness and cross-orthogonality per row/vertex.
- Mode II — `_check_fusion_closure` (cyclic), `_check_nested_flag_closure`,
  `_check_matrix_unit_closure`, `_check_fusion_closure_general`
  (group), plus `_check_flat_spectrum`/`_check_group_boundary` for the
  boundary vector and `_check_channel_span_full_rank`/
  `_check_group_representation` for the channel/representation structure
  itself.
- Mode II-B — `verify_intertwiner`: direct residual of the intertwiner
  condition (null-complement and boundary-compatibility conditions of the
  thesis are not checked).

On top of that, every assembled chain goes through the numerical
safeguards described in [Background](#background) (normalization, and the
exact dense check for $d^N\le2^{10}$), regardless of which per-mechanism
certificate was used to build the tensor. The test suite (`tests/test_mpu_generators.py`) mirrors this same
two-layer structure and additionally builds fully dense $d^N \times d^N$
matrices for direct $U^\dagger U = \mathbb{1}$ checks at small $N$, as an
independent cross-check of the whole pipeline, not just of the
certificate functions in isolation.

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
- `TreeMPU.checks.dense_unitary_from_tensors` — the exact dense check.

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
explicit negative-control regression `test_naive_sum_without_capped_site_fails`.
`tests/test_certificates_and_api.py` checks that every algebraic
certificate rejects a perturbed input, the rectangular Lee-Yang
intertwiner, the Lee-Yang boundary family, the family registry and
reproducibility.

## Reference

The theory implemented here is derived in the author's master's thesis,
*Beyond-QCA MPUs: Generation, Certification, and Circuit Synthesis*
(M. Vassallo, University of Bologna, 2026; [AMS Laurea](https://amslaurea.unibo.it/)
once published), and in the manuscript "Tensor-First Sampling of Exact
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

Alpha (0.2.0). Used as a submodule of
[`TreeMPU`](https://github.com/Mrcocvs02/TreeMPU). All
mechanisms listed above are implemented and covered by the test suite.

## License

MIT — see [LICENSE](LICENSE).
