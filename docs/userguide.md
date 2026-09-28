# mpu_generators user guide

`mpu_generators` samples **exact** open-boundary Matrix Product Unitaries:
every returned tensor chain is unitary for every system size $N$, certified
by an algebraic identity of the local data rather than by testing finitely
many $N$. It implements the generative mechanisms of the thesis chapter
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
$V,W$ and $P_i$ projectors on balanced column blocks of $W$.

**Mode II — fusion closure** (`fusion`). $A[a,b]=\delta_{ab}y_a$, so
$U_N=\sum_u\ell_u y_u^{\otimes N}$. Unitarity for every $N$ follows from
$y_u^\dagger y_v\in\{0\}\cup\{y_w\}$ plus a boundary on the skeleton's
admissible manifold:

| skeleton | admissible boundary |
|---|---|
| `group` (left-regular representation, any listed finite group) | Peter–Weyl manifold $\prod_\rho U(d_\rho)$ |
| `cyclic` ($\mathbb Z_M$ characters; group convolution iff $M\mid d$) | flat Fourier spectrum |
| `nested_flag` (contains MCZ) | flat torus $\ell_w=-S_w+e^{i\theta_w}$ |
| `matrix_units` (contains subspace rotations) | $\ell_{(i,j)}=(V-\mathbb 1)_{ij}$, $V\in U(m)$ |

Each site is dressed by an independent Haar unitary $y\mapsto WyW^\dagger$
(or one shared $W$ for the translation-invariant generator).

**Mode II-B — bond-resolved fusion** (`fusion_bond_resolved`). Channels
are small MPOs; non-monomial rules (Lee–Yang $\sigma\cdot\sigma=e+\sigma$)
are certified by an injective intertwiner $B_{\sigma\sigma}V=V(B_e\oplus B_\sigma)$
(`skeletons.solve_intertwiner`). Outside the scope of the paper.

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
* `verify=True` (default) runs the algebraic checks and, for $d^N\le2^{10}$,
  an exact dense unitarity check.
* **Reproducibility:** each call reseeds from `seed`; for uniform generators
  the site tensor does not depend on `n_sites`.
* `build_family(name, …)` reproduces the thesis/paper instances with their
  production settings.
* `mixing.stack` (operator product) and `mixing.capped_sum` (Mode I readout
  of two chains) combine certified chains.

## 5. Conventions and caveats

* Site order: construction order is reversed at the end, so with per-site
  resampling, chains for increasing $N$ extend a common sequence at the
  left end (Mode II-B: at the right end).
* The single-tensor translation-invariant constructions of
  `UniformMPUGenerator` (`clock`, `self_direction`, `resonant`,
  `multi_symbol`, `invariant_subspace`, `degenerate_rows`) come from working
  notes and are not part of the thesis or paper; `self_direction`,
  `resonant` and `multi_symbol` give on-site product unitaries.
* Group convolution is implemented for abelian groups and $S_3$ (the paper
  treats every finite group).

See the tutorials in `../../notebooks/` (02, 03, 06) and the theory-to-code
map in `../../docs/theory_code_map.md`.
