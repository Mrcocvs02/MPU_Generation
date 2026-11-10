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

## 4. Choosing a generator

Every construction is exactly unitary for every $N$ (Mode II-B: validated
numerically). What differs is the structure you get. Start from what you want:

| you want | use | bond dim | what to expect |
|---|---|---|---|
| a generic MPU beyond QCA, one fixed rule for all $N$ | `branching` (Mode I), `build_family("branching_uniform")` | any $D$ | one-directional operator spreading that **decays** with the distance; first-cut operator-Schmidt rank $\le d$ |
| **long-range, non-decaying** operator spreading, analytically transparent | `shared_basis_chain` (ring or permutation maps), `cayley_chain` | any $D$, also $D>d$ | the weight outside any window does not shrink with the radius; one direction only; **not** from Mode I in general (see below) |
| the same, translation invariant | `UniformMPUGenerator.shared_basis_chain` / `shared_basis_automaton` | any $D$ | as above, one tensor reused |
| a rule that depends on a **global** property of the input, at fixed bond | `multi_start`, `build_family("multi_start_parity")` | $m^2D$ | several starts; beyond any single start at fixed bond; section 4.3 |
| a charge-conserving phase / permutation function (reflection symmetric) | `fusion(skeleton="cyclic")` (also $M>d$) | $M$ | $R\,e^{i\varphi(Q \bmod M)}R^\dagger$; symmetric, constant tail |
| an MCZ-type diagonal-phase family | `fusion(skeleton="nested_flag")` | $\le d$ | contains MCZ |
| group convolution (abelian or $S_3$) | `fusion(skeleton="group")` | $d=\lvert G\rvert$ | regular representation, dressed by a Haar unitary |
| first-cut rank above $d$ (matrix units) | `fusion(skeleton="matrix_units")`, `grafted` | $1+m^2$, $2n_{ch}$ | outside Mode I, see below |
| a fusion-category MPU (Lee-Yang) | `fusion_bond_resolved(lee_yang_skeleton())` | 14, $d=5$ | unitarity validated numerically only |
| the instances of the thesis / paper | `build_family(name, ...)` | per family | same settings as the numerical chapter |
| site-dependent bases, maps, bond dimensions | `MPUGenerator` (any method); `shared_basis_chain(R=[...], V=[[...]], D=[...])` | per site | nothing couples the bases of different sites |

`TreeMPU` synthesizes all of them; its cost depends on $(d,\chi,\ell)$ and on the
conditioning $q$, not on the operator spreading (a ring with Haar bases and one with a shared
basis give identical $q$, $\ell$, qubits, depth and gates with the closed-form
Mode I boundary, `boundary_mode1_exact=True`). Section 4.4 lists which families
also have a known structured circuit (all but Lee-Yang).

### 4.1 Operator spreading: what keeps it, what destroys it

Measured as the fraction of $U_N P_0 U_N^\dagger$ ($P\in\{X,Z\}$, worst of
the two) not supported in a window of radius $R$ around site 0. All numbers
are finite-size numerical observations, not theorems.

*Why the shared-basis chain does not decay.* At every site all vertices read
the input in one orthonormal basis, so $U|r_{x_1}\cdots r_{x_N}\rangle$ is a
product state whose site $k$ depends on the letters on its left only through
the vertex $v_{k-1}$, a deterministic function of them. With permutation maps
no information is lost along the path, so a changed letter changes every
later output. It is a finite-state (group) automaton with a coherent,
basis-diagonal memory; the whole operator is still a coherent entangling
unitary (operator-Schmidt rank $D$, arbitrary local unitaries).

*What it costs to let vertices read differently.* A path that passes through
vertices reading in different projector sets scrambles the record, and the
tail decays. Ring, $d=2$, $N=8$, drop of the weight between $R=1$ and $R=3$
(about 0 is flat), 8 seeds:

| choice of bases | drop |
|---|---|
| one basis for all vertices (new independent basis at every site) | 0.00 |
| same basis, letter labels swapped on some vertices | 0.00 |
| per-vertex rotations $R_0e^{i\theta\,n_v\cdot\sigma}$, $\theta=0.1/0.2/0.4/0.8$ | 0.02 / 0.05 / 0.12 / 0.16 |
| two mutually unbiased bases alternating | 0.16 |
| every vertex in its own Haar basis (`branching`) | 0.16 |
| one vertex of four in another Haar basis | 0.09 |

Rules that follow (all from these runs):

* **Free** (keeps the tail flat): a new reading basis at every site, any
  vertex unitaries $V_{k,v}$, any permutation of the letter labels per vertex,
  any bond dimensions, any permutation maps.
* **Costly**: a vertex that reads in a different *set of projectors*. The loss
  grows with the angle and with the number of such vertices; it saturates near
  the Haar value.
* **Position matters**: mixing near the source costs most. Mixed sites at
  1-2 reduce the weight outside radius 3 from 0.82 to 0.53; at 4-5 to 0.76; at
  the last two sites nothing ($N=9$, $D=4$).
* **No recovery was found, and later shared sites have no leverage.** Shared-basis
  sites after a mixed segment do not restore the weight (0.53 stays 0.53), and
  further mixing only lowers it. The weight outside the window is independent
  (to about $2\times10^{-4}$, $N=9$) of the reading bases and of the vertex
  unitaries of the later shared sites, which is why a numerical optimisation of
  those bases (L-BFGS, six sites, two seeds) moved nothing (0.31 to 0.31). The
  operator norm is conserved, so the lost weight sits elsewhere (inside the
  window or to the left); the far sites do not get it back. The only choices
  found to change the weight are the projector sets of the vertices, and
  those lowered it. Not searched: other maps $\sigma$ at the later sites.
* Non-injective maps (random maps) coalesce paths and also decay; random
  *permutations* can have an exactly zero tail when the two letters lead from
  the start vertex to the same vertex.

### 4.2 What Mode I cannot represent

For a Mode I chain with a single start vertex (`readout=None`) at most $d$
branches leave the first site, so the operator-Schmidt rank of $U$ across the
cut (site 1 | rest) is at most $d$. This is a necessary condition, easy to
check with `TreeMPU.checks.dense_unitary_from_tensors`. Computed values
(first-cut rank; $d$ in brackets):

| unitary | rank | in Mode I? |
|---|---|---|
| `branching` $D=4$ ($d=2$), $D=3$ ($d=3$) | 2, 3 | yes (by construction) |
| shared-basis ring $D=4$, Cayley $S_3$ ($d=2$) | 2, 2 | yes (special cases) |
| Mode II `cyclic` $M=2,3$; `cyclic` $M=3>d=2$; `nested_flag`; `group` $S_3$ | $\le d$ | not excluded (`cyclic` with $M=d$ is a shared-basis chain; `nested_flag` is one deterministic chain, section 4.3) |
| Mode II `matrix_units` ($d=3,\chi=5$) | 5 > 3 | **no** |
| `grafted` ($d=2,\chi=4$) | 4 > 2 | **no** |
| Mode II-B Lee-Yang ($d=5,\chi=14$) | 13 > 5 | **no** |
| SWAP of two qubits (shift-type) | 4 > 2 | **no** |
| generic depth-2 brickwork QCA, Haar 2-qubit gates | 4 > 2 | **no** |

So generic finite-depth QCAs and shift-type operators are outside Mode I (and
outside every shared-basis chain), as are the Mode II constructions with
larger bond dimension. The families complement each other; none contains the
others. A rank $\le d$ does not prove membership.

### 4.3 More than one start vertex

*New, not in the thesis.* Mode I certifies a chain that leaves one start vertex
(`l_vec` a single basis vector). Other constructions do start from several
vertices; this section says what is known and how to build them on the same
$N$ sites. Adding control sites is not an alternative: the controls end up
entangled with the system (for a cascade with $N=4$ and two controls, 1.49 of
the 2 possible bits), so the result is not a unitary on the $N$ sites.

**The number of start vertices is not gauge invariant.** The triangular tensor

$$A[0,0]=s\,T b,\quad A[0,1]=c\,T e,\quad A[1,1]=T,\qquad
s^2+c^2=1,\ \{b,e\}=0,\ b^2=e^2=1,$$

with $T$ unitary, gives $U=T^{\otimes N}\Gamma_N$, where
$\Gamma_N=c\,e\otimes 1+s\,b\otimes\Gamma_{N-1}$, $\Gamma_0=1$, is a reflection
(a Majorana-string operator for $b=Z$, $e=X$). The same tensor with
$\ell=(i\sin\theta,\cos\theta)$ gives $T^{\otimes N}e^{i\theta\Gamma_N}$: two
start indices. Yet both are Mode I chains with $D=2$ and one start vertex in a
*site-dependent* gauge: the vertices at level $k$ are $\pm\theta_k$ with
$\sin\theta_{k+1}=s\sin\theta_k$, and the projectors are
$(1\pm bP/\rho)/2$ with $P=\cos\theta+i c\sin\theta\,e$ and $\rho=\lVert P\rVert$
(checked against the triangular form to $10^{-15}$, also with $s$, $T$, $b$, $e$
changing from site to site). So a uniform tensor that fails (RR) can still be
Mode I. What is gauge invariant is the first-cut rank of section 4.2: a rank
$>d$ excludes a single start in any gauge.

**Multi-start Mode I (`multi_start`).** Mode I decides left to right, so it
cannot act on what it has not read yet. Several starts lift this: each block
*guesses* a global property of the input and the boundary keeps the right guess,
as a nondeterministic automaton does. Take one shared-basis automaton and give
letter $a$ a charge $q_a\in\mathbb Z_m$. The copy $(\chi,g)$ has edges

$$A_k^{(\chi,g)}[v,\sigma_{k,a}(v)]\mathrel{+}=e^{i\theta_{k,\chi}(v,a)}\,\omega^{gq_a}\,
V_{k,v}\lvert r_{k,a}\rangle\langle r_{k,a}\rvert,\qquad
c_{\chi,g}=e^{i\phi_\chi}\omega^{-\chi g}/m ,$$

with start weight $c_{\chi,g}$. All copies follow the same path, and the sum over
$g$ gives $\delta_{\chi,q(x)}$, so $U_N\lvert r_x\rangle=e^{i\phi_{q(x)}+i\Theta_{q(x)}(x)}
W_N\lvert r_x\rangle$: the plain automaton times a diagonal unitary, for every $N$ and
all data. Every block is a Mode I chain. The start weights are not free: one phase
per sector, nothing else.

The "guess" depends on the gauge. A Fourier transform of the index $g$ leaves $U$
unchanged and labels the copies by the running charge instead; then every
reading-basis input has exactly one accepted run (checked for $m=2,3$, $N=5$, all
inputs). So `multi_start` is an *unambiguous* automaton with phases, and it is
built from two deterministic ones, one reading from each side (section 4.4).

```python
from mpu_generators import UniformMPUGenerator, build_family, mixing

chain = UniformMPUGenerator(seed=0).multi_start(n_sites=6, D=3, transitions="ring")
parity = build_family("multi_start_parity")        # smallest example, bond 4
wider = mixing.stack(                              # first-cut rank above d
    UniformMPUGenerator(seed=1).branching(d=2, bond_dim=2, n_sites=6), chain
)
```

How it relates to the rest:

| setting | what it is |
|---|---|
| `n_sectors=1` | the counter, `shared_basis_chain` |
| `edge_phases=None`, `D=1`, $V=1$ | Mode II `cyclic`: phases on the sectors of $y^{\otimes N}$, also $m>d$ |
| `edge_phases=None` | $W\cdot C$: the counter times Mode II |
| edge phases | new: the guess carries its own phase along the path |
| `mixing.stack(w, chain)`, $w$ reading in another basis | first-cut rank up to $d^2$; the blocks are products of chains, no longer Mode I |

**Mode II is only partly inside.** A Mode II chain whose channels are unitary is $M$
trivial Mode I chains of bond 1 in parallel: true for `cyclic` and `group`, and
for `grafted` with blocks of bond 2 (checked block by block), though `multi_start`
itself covers only $\mathbb Z_m$ charges. In `nested_flag`, `matrix_units` and Lee-Yang most blocks
are not unitary (rank 1 in `matrix_units`), so the blocks are not Mode I chains.
For `matrix_units` and Lee-Yang the whole operator is outside too (first-cut rank
$>d$, section 4.2). `nested_flag` is not: the whole operator is one deterministic
chain with a single start and one vertex per channel, in the shared basis (equal
up to a global phase within $4\cdot10^{-14}$; 30 uniform and 30 site-dependent
chains; not a package test). Unitary blocks are also not enough: the cross terms need a condition
between the blocks (two independent Mode I blocks with weights
$\cos t,\ i\sin t$: median error 0.47 over 200 draws).

**Why it is more than a single start.** With $m=2$, $D=1$, $\theta_0=0$,
$\theta_1(a)=a\theta$, the operator is $P+k^{\otimes N}Q$, $k=\mathrm{diag}(1,e^{i\theta})$,
$P,Q$ the parity projectors: bond 4 at every $N$. A single start needs at least
$j+1$ vertices at level $j$ (generic $\theta$). The first-cut operators span the
diagonal matrices, so the reading basis is forced, and a vertex is then determined
by the operator left to act, up to a phase. After a prefix with $n$ ones this
depends on $n$: $j+1$ classes, counted exactly up to $N=12$. So no fixed-bond Mode I
chain gives this family. This is about size, not rank: the first-cut rank is
still at most $d$.

**Cost and limits.** The bond is $m^2D$, $m$ times less without edge phases, plus
$mDd$ phases per site. It is minimal: the largest operator-Schmidt rank over the
cuts, a lower bound for every MPO of the operator, reaches $m^2D$ ($d=2$,
site-dependent data: $m=3$, $D=1$ from $N=8$; $m=3$, $D=2$ from $N=12$; $m=4$, $D=1$
from $N=10$; uniform data need longer chains, $m=3$ from $N=16$). Without edge
phases it is $mD$. The cost that matters is the conditioning: with `TreeMPU` the
circuit depth follows $q$. At $\chi=4$, $L=3$: counter $q\approx4.7$, depth 44;
`multi_start` with random phases $q$ from 10 to 187, depth from 84 to 1474, because
two guesses can come close; with chosen phases $q=5.4$, depth 44, as in
`multi_start_parity`. Draw phases only if you can afford this. Measured at $N=256$, $D=4$: 78 ms against 38 ms for the counter of
the same $D$ and 116 ms for a counter of the same bond 16 (`verify=False`). The
operator-Schmidt rank grows by 2 per site up to $4D$ on an odd ring, so
Assumption 1 needs blocks of about $2D$ sites ($L=6$ for $D=3$). If the vertex already
determines the charge the Fourier blocks are redundant: on an even ring with $m=2$
the rank is $2D$, the representation is not minimal, and Assumption 1 fails at every
blocking. Weight of $X_0$ outside radius 1, 2, 3 at $N=8$, 6 seeds: `multi_start`
ring $D=3$ 0.68, 0.68, 0.68 (the counter 0.83, flat), so the long tail survives.

**Other constructions that exceed the single-start bound.** All on $N$ sites, all
proved unitary. Each is a direct sum of disjoint sub-digraphs, one start
(at least) in each.

| construction | blocks | unitary because | first-cut rank ($d=2$) | how to build it |
|---|---|---|---|---|
| grafted | $M$ copies $K_N y_g^{\otimes N}$, each unitary | $U=K_N\sum_g c_g y_g^{\otimes N}$, a Mode I factor times a Mode II one | 4 | `grafted` |
| $W\cdot C$ | one block per channel of $C$, each unitary | product of two unitaries | 4 | `mixing.stack(w, c)` with `w` any chain, `c` a `fusion` chain |
| two unitary blocks | $V\cos t$ and $iVS\sin t$ | $S$ a reflection; $S$ need not be a product (MCZ, $XZ^{\otimes N}X^\dagger$ and $\Gamma_N$ tested) | not computed | not implemented |
| sector-controlled | $VP$ and $VKQ$, not unitary | $P+Q=1$ projectors, $[K,P]=0$ | 4 | not implemented |
| Householder | $V$ and $V\lvert\psi\rangle\langle\psi\rvert$, rank one | $Q=\lvert\psi\rangle\langle\psi\rvert$, $K=e^{i\varphi}$; $\psi$ any MPS | 4 | not implemented |

The last four rows have one form, $U=V(P+KQ)$. For two unitary blocks the form
is forced: writing $U=U_1(c_1+c_2Z)$ with $Z=U_1^\dagger U_2$, unitarity makes
$Z$ a phase times a reflection, hence $U=U_1e^{i\theta S}$ (proved). Mode II with
two sectors is the case where $P,Q$ are symmetry projectors and $K$ a phase; in the
sector-controlled form the $K$ of different sectors are unrelated, for instance
$P=(1+Z^{\otimes N})/2$ and $K$ a brickwork of parity-conserving gates.

Most of this table is the grafted mechanism in a wider form. $W\cdot C$ is
`grafted` with the two-vertex path replaced by any chain and the cyclic channels
by any `fusion` skeleton. A sector form with $K$ a phase is $W\cdot C$: removing
a factor $e^{i\theta y^{\otimes N}}$ brings the first-cut rank back to $d$
(excess weight $10^{-22}$). Householder with product $\psi$ is $W\cdot C$ with
channels $1$, $\lvert\psi\rangle\langle\psi\rvert^{\otimes N}$, and an entangled
$\psi=G\lvert 0\dots0\rangle$ only adds a conjugation by the sequential unitary
$G$. Likewise $\Gamma_N$ and $XZ^{\otimes N}X^\dagger$ give products of chains that
are already available. What is not reached this way is the sector form with $K$
not a phase: the same search over $y$, $\theta$ leaves an excess of 0.002 to 0.37
($K$ product or brickwork; two channels and unitary $y$ only, so this excludes
that form and is not a general proof). There the blocks cancel by orthogonality,
not by fusion closure: $K$, $K^2$, ... need not close on a finite set.

Constraints of the sector form with $P=(1+y^{\otimes N})/2$: $y^2=1$; $K_N$ unitary
and commuting with $y^{\otimes N}$ at every $N$ (for unitary $K$ this is also
necessary); start weights $(\tfrac12,\tfrac12)$ on $1,y^{\otimes N}$ and
$(\tfrac12,-\tfrac12)$ on $K,Ky^{\otimes N}$, up to one free phase per sector. Other
weights break unitarity (error 0.2 for magnitudes 0.55, 0.45; 1.4 for equal signs).
With these, a uniform chain with fixed tensors and boundaries works for every
$N$: $K$ a staircase of one parity-conserving gate passes the exact test below
($\dim W=24$; with a ring $V$, 76).

Any unitary can also be written $U=H+iG$ with $H=(U+U^\dagger)/2$ and
$G=(U-U^\dagger)/2i$. Hermiticity alone does not cancel the cross term, which is
$i[H,G]$: one needs $[H,G]=0$ and $H^2+G^2=1$, automatic for a unitary. As an ansatz
with $H$ and $G$ in two disjoint blocks of small bond it only found $U$ with few
distinct eigenvalues (at most 8 at $N=6$; first-cut rank 3 at most).

```python
from mpu_generators import UniformMPUGenerator, mixing

w = UniformMPUGenerator(seed=3).shared_basis_chain(n_sites=6, d=2, D=4, transitions="ring")
c = UniformMPUGenerator(seed=4).fusion(d=2, bond_dim=2, n_sites=6, skeleton="cyclic")
tensors, l_vec, r_vec = mixing.stack(w, c)   # D=8, two components, two start vertices
```

Measured weight outside radius $R=1,2,3$ of $X_0$ ($N=8$, 6 seeds): shared-basis
ring 0.83, 0.83, 0.82; $W\cdot C$ with `cyclic` 0.91, flat. So a chain with several
starts can keep the non-decaying tail. The sector-controlled form gives about
1.0: $X_0$ flips the global parity, hence the sector.

**An exact test for any digraph.** Let $B[p,p']$ be the site tensor of
$U^\dagger U$ on the doubled bond, $L=\bar\ell\otimes\ell$,
$R=\bar r\otimes r$, and $W$ the span of $L\,B[w]$ over words $w$ of length
$\ge n_0$. A uniform chain is unitary for every $N\ge n_0$ if and only if
$U_{n_0}$ is unitary and

$$w\,B[p,p']\,R=\delta_{pp'}\,w\,R\qquad\text{for all } w\in W.$$

$\dim W\le D^2$, so this is a finite check (not implemented in the package). It
also says what any constraint on digraph and local terms has to do: fix a
subspace $W$ closed under $B$ on which the condition holds. Mode I takes the
diagonal $e_v\otimes e_v$. With disjoint blocks, $W$ has a sector for each pair
of blocks, and the cross sectors have to cancel: in pairs (anti-Hermitian
cross terms: grafted, reflections) or one by one (orthogonal blocks:
sector-controlled).

The dimension of the same orbit at a cut, computed at that cut with the
minimal bond, is a gauge-invariant lower bound on the vertices of a Mode I
representation: `grafted` has $\dim W_k=6>D_k=4$, $W\cdot C$ 14 > 8.

**How far the classes go.** A numerical search over uniform chains with disjoint
blocks ($d=2,3$, bonds up to 3+3, 6150 starts, no block assumed unitary) found
192 solutions with at least two blocks, each certified for every $N$ by the test
above. Those with two unitary blocks
all have $Z$ a reflection. Six have first-cut rank $>d$. Four are a unitary
block plus a block of rank 1 or 2, hence Householder (a unitary that differs
from $V$ by low rank has this form). One has orthogonal blocks $UP$, $UQ$ with
$2P-1$ a reflection of bond 2, not a product. The last is two copies of one
bond-2 tensor with different boundaries and an on-site reflection on the input
legs; neither copy is unitary and no factorization was found. A finite search:
this is not a classification.

Reversing a chain (swap the bond legs, reverse the site order, exchange
`l_vec` and `r_vec`) gives the column-resolved mirror: all vertices are starts
and one vertex is the end. Composing forward and mirror chains gives a tail
in both directions (weight beyond two sites, $N=9$, $D=3$, middle site:
forward 0.40 left, 0.75 right; mirror 0.79, 0.37; product 0.87, 0.75).

### 4.4 Known circuit structure

*New, not in the thesis.* Besides the generic tree circuit of
[arXiv:2508.08160](https://arxiv.org/abs/2508.08160) (what `TreeMPU` builds),
almost every family here has an explicit structured circuit: staircases that
compute a label into ancillas, act, and erase it. Ex. 14 and Ex. 24 refer to
Styliaris, Trivedi, Pérez-García, Cirac, Quantum 2025
([arXiv:2406.10195](https://arxiv.org/abs/2406.10195)). The circuits are built
site by site from the site tensors, so they also hold for site-dependent data.
Keep this in mind when an MPU is the target of a circuit ansatz: a known
structured circuit makes it a special target. The circuits use ancillas, so they
do not bound the depth of an ancilla-free ansatz.

| family | structured circuit | ancillas | checked against the tensors |
|---|---|---|---|
| `branching` (any Mode I chain) | forward staircase: ancilla $k$ gets a coherent copy of the branch taken at site $k$, controlled by ancilla $k-1$; last gate; backward: erase the copies, apply $V_v$ | $N-1$, dim $D_k$ | $\le 6\cdot10^{-15}$ (8 cases) |
| counter (`shared_basis_chain`, `cayley_chain`), `nested_flag` | Ex. 24 (deterministic weighted automaton): forward staircase computes the vertex, backward erases it; plus the basis change $R_k$ and $V_{k,v}$ on site $k$ | $N-2$, dim $D$ | `nested_flag` rewritten as one chain ($4\cdot10^{-14}$); counter: not checked separately |
| `multi_start` (also `cyclic`: `D=1`, no edge phases) | two-sided: forward staircase (vertex, prefix charge), backward staircase (suffix charge), one layer of local phase gates, both erased | one of $mD$ and one of $m$ values per cut | $\le 3\cdot10^{-15}$ (also site-dependent) |
| `group` (regular representation) | $U=C^\dagger(L\otimes 1)C$; $C$: $h_k\mapsto h_{k-1}^{-1}h_k$, a staircase of classical reversible 2-site gates; $L=\sum_g\ell_g\rho(g)$ on one site | none | $\le 3\cdot10^{-15}$ ($\mathbb Z_3$, $\mathbb Z_2^2$, $S_3$) |
| `matrix_units` | Ex. 14: $U-1$ has rank $m$ on a span of product vectors; a fan-out maps it to the first site, then a gate on site 1 controlled by the rest | as Ex. 24 | range of $U-1$: $\le 2\cdot10^{-15}$ |
| `grafted` | $U=U_KU_{\rm cyc}$: a two-vertex Mode I chain times a `cyclic` chain | of the factors | $\le 1.4\cdot10^{-15}$ |
| `mixing.stack`, `multi_start_stacked` | products of the above | of the factors | by construction |
| sector form $P+KQ$ (not implemented) | compute the sector into an ancilla, apply $K$ controlled by it, erase | 1 | $1.4\cdot10^{-16}$ |
| Lee-Yang | **none known** | | |

The checks are numerical scripts outside the package, on every reading-basis
input.

The Lee-Yang operator is $U=1+(u_e-1)O_e+u_\sigma O_\sigma$, with fusion-category
MPOs $O_e,O_\sigma$. It has three eigenvalues $1,e^{i\alpha},e^{i\beta}$, and it is
not $C^\dagger(L\otimes 1)C$ for any unitary $C$ (proved: that form has eigenspace
dimensions divisible by $5^{N-1}$). The paper above says that this class
(MPO-projector unitaries) "lacks a known quantum circuit implementation", apart
from its generic construction. It is the only package family without a structured
circuit. It acts non-trivially on a vanishing fraction of the space: $7/25$,
$18/125$, $47/625$ at $N=2,3,4$ (Lucas numbers, $\approx(\varphi^2/5)^N$). So its
distance from the identity goes to 0 in the normalized Frobenius norm. In the
operator norm it does not: it is 2 at $(\alpha,\beta)=(0,\pi)$.

## 5. API in brief

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

## 6. Conventions and caveats

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
