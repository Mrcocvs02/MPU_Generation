# mpu_generators user guide

`mpu_generators` samples open-boundary Matrix Product Unitaries that are
unitary for every system size $N$: for Modes I and II and the grafted
composition, unitarity follows from an algebraic identity of the local data
(proved in the thesis) rather than from testing finitely many $N$. The
exception is Mode II-B (Lee–Yang), whose unitarity is validated numerically
only. The package implements the generative mechanisms of the thesis chapter
"Generating MPUs" and the tensor-first sampling framework (M. Vassallo,
"Tensor-First Sampling of Exact Matrix-Product Unitaries", 2026).

## 1. Why

Benchmarking circuit synthesis, operator spreading or learning algorithms
needs large ensembles of exact many-body unitaries with controlled causal
structure. Circuit-first constructions fix an architecture and inherit its
light cone. Here the probability measure is placed directly on certified
local tensor or boundary data (*tensor first*), with one rule valid at
every $N$, reaching MPUs beyond any quantum cellular automaton.

## 2. Installation

```bash
pip install -e . -e ./mpu_generators      # from the TreeMPU repository root
pytest mpu_generators/tests               # ~30 s
```

## 3. Certification mechanisms

**Mode I — row resolution** (`branching`). The bond indexes the vertices of
a digraph; each row satisfies
$\sum_u A[v,u]^\dagger A[v,u]=\mathbb 1$, $A[v,u]^\dagger A[v,u']=0$
($u\ne u'$) and the mirror identities. With $\ell=e_0$, $r=(1,\dots,1)$ the
chain is unitary for every $N$. Sampler: $A[v,u_i]=VP_iW^\dagger$ with Haar
$V,W$ and $P_i$ projectors on consecutive column blocks of $W$. By default
the digraph is random with out-degree $\min(d,\chi)$ and the block sizes
(ranks) are balanced; `digraph=`, `out_degree="random"` and
`ranks="random"` or `ranks={v: [...]}` select other ensembles on the same
certified class (the certificate holds for every rank partition).
`MPUGenerator(coherent_in_n=False)` draws an independent family for every
chain length instead of one family for all $N$.

**Mode II — fusion closure** (`fusion`). $A[a,b]=\delta_{ab}y_a$, so
$U_N=\sum_u\ell_u y_u^{\otimes N}$. Unitarity for every $N$ follows from
$y_u^\dagger y_v\in\{0\}\cup\{y_w\}$ plus a boundary on the skeleton's
admissible manifold:

| skeleton | admissible boundary | constraints |
|---|---|---|
| `group` (left-regular representation; `Zn`, `Zn_product`, `S3` implemented) | Peter–Weyl manifold $\prod_\rho U(d_\rho)$ | `d = bond_dim = \|G\|` |
| `cyclic` ($\mathbb Z_M$ characters; group convolution iff $M\mid d$) | flat Fourier spectrum | `bond_dim <= d` |
| `nested_flag` (contains MCZ) | flat torus $\ell_w=-S_w+e^{i\theta_w}$ | `bond_dim <= d` |
| `matrix_units` (contains subspace rotations) | $\ell_{(i,j)}=(V-\mathbb 1)_{ij}$, $V\in U(m)$ | `bond_dim = 1 + m**2`, `m <= d` |

For `grafted`, `bond_dim` is even with `bond_dim / 2 <= d` (the channels are
the `cyclic` skeleton).

Each site is dressed by an independent Haar unitary $y\mapsto WyW^\dagger$
(or one shared $W$ for the translation-invariant generator).

**Mode II-B — bond-resolved fusion** (`fusion_bond_resolved`). Channels
are small MPOs; non-monomial rules (Lee–Yang $\sigma\cdot\sigma=e+\sigma$)
are described by an injective intertwiner $B_{\sigma\sigma}V=V(B_e\oplus B_\sigma)$.
`skeletons.solve_intertwiner` and `skeletons.verify_intertwiner` are
standalone helpers for this condition; the generators do not call them, and
`lee_yang_skeleton` builds its tensors explicitly. **No algebraic certificate
is checked at generation: the unitarity of Mode II-B chains is validated
numerically only** (normalization for any $N$, and a dense check to tolerance
for $d^N\le2^{10}$). Outside the scope of the paper.

**Grafted composition** (`grafted`): a Mode I two-vertex path inside Mode II
channels, $U_N=K_N\,U_N^{\rm conv}$.

## 4. API in brief

```python
from mpu_generators import MPUGenerator, UniformMPUGenerator, build_family, FAMILIES

tensors, l_vec, r_vec = UniformMPUGenerator(seed=0).branching(d=2, bond_dim=2, n_sites=8)
tensors, l_vec, r_vec = MPUGenerator(seed=1).fusion(
    d=3, bond_dim=3, n_sites=6, skeleton="group", group="Zn", group_n=3
)
tensors, l_vec, r_vec = build_family("fusion_nested_flag", d=2, bond_dim=2, seed=4, n_sites=16)
```

* Every method returns `(tensors, l_vec, r_vec)` with tensors
  `A[p_in, p_out, bond_L, bond_R]`; pass `tensors[0]` (uniform) or `tensors`
  (site dependent) to `TreeMPU.UniformMPU` / `GeneralMPU`.
* `n_sites` or `n_blocks`; `blocking=L` merges L sites per tensor;
  `blocking="auto"` finds the smallest block size satisfying Assumption 1.
* `verify=True` (default) runs the algebraic checks that exist for the mode
  (the boundary condition is checked only for `cyclic` and `group`, not for
  `nested_flag` and `matrix_units`; none for Mode II-B), then the numerical
  safeguards: normalization for any $N$ and, for $d^N\le2^{10}$, a dense
  unitarity check to tolerance ($\|U^\dagger U-\mathbb 1\|_F\le10^{-8}\sqrt{d^N}$).
* `branching(readout=...)` appends $m$ sites, so `blocking` must divide both
  `n_sites` and `n_sites + m` (hence $m$).
* **Reproducibility:** each call reseeds from `seed`; with
  `coherent_in_n=True` (default) the site tensor of uniform generators does
  not depend on `n_sites`.
* `build_family(name, …)` reproduces the thesis/paper instances with their
  production settings.
* `mixing.stack` (operator product) and `mixing.capped_sum` (Mode I readout
  of two chains) combine certified chains.

## 5. Conventions and caveats

* Tensors are `A[p_in, p_out, bond_L, bond_R]`: the first physical index is
  the input. The block stored at `A[:, :, v, u]` is the transpose of the
  operator $A[v,u]$ of the thesis. Boundary vectors contract linearly,
  without complex conjugation. Dense matrices from
  `TreeMPU.checks.dense_unitary_from_tensors` are `U[out, in]` with site 0
  least significant.
* Site order: construction order is reversed at the end
  (`_finalize_chain`), so with per-site resampling, chains for increasing $N$
  extend a common sequence at the left end (Mode II-B: not reversed, at the
  right end). After blocking, the list holds one tensor per block.
* The single-tensor translation-invariant constructions of
  `UniformMPUGenerator` (`clock`, `self_direction`, `resonant`,
  `multi_symbol`, `invariant_subspace`, `degenerate_rows`) are not part of
  the thesis or paper and are validated by the unit tests only;
  `self_direction`, `resonant` and `multi_symbol` give on-site product
  unitaries.
* Group convolution is implemented for `Zn`, `Zn_product` (abelian) and `S3`;
  the construction and the paper apply to every finite group, given its
  tables.

See the [tutorials](https://github.com/Mrcocvs02/TreeMPU/tree/main/notebooks) (02, 03, 06) and the
[theory-to-code map](https://github.com/Mrcocvs02/TreeMPU/blob/main/docs/theory_code_map.md) in the TreeMPU repository.
