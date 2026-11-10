# Changelog — mpu_generators

## 0.3.0 (2026-10-22)

All default outputs are unchanged (bit for bit).

### Documentation
- `docs/userguide.md` section 4 "Choosing a generator": a table by goal, what
  keeps or destroys long-range operator spreading (measured), and what Mode I
  cannot represent (first-cut operator-Schmidt rank criterion), and how to
  build chains with several start vertices on the same sites (section 4.3),
  including the minimal bond of `multi_start`; section 4.4 "Known circuit
  structure": the explicit circuit of every family (none known for Lee-Yang).
  Corrected: `nested_flag` as a whole is a single-start Mode I chain (only its
  blocks are not).

### Changed
- Internal: every mode is now a `Skeleton` subclass (`mpu_generators.skeletons`)
  driven by one pipeline, `MPUGenerator.sample(skeleton, ...)`; the generator
  methods are thin wrappers. `FusionSkeleton` is kept as an alias of
  `BondResolvedSkeleton`. All outputs are unchanged bit for bit (golden
  fingerprints in `tests/test_golden_outputs.py`).

### Added
- `MPUGenerator.multi_start` (also on `UniformMPUGenerator`) and
  `skeletons.MultiStartSkeleton`: Mode I with several start vertices, copies of
  one shared-basis automaton, one per guess of a total charge. Four new entries
  of `FAMILIES`: `multi_start_parity`, `multi_start_ring`, `multi_start_general`,
  `multi_start_stacked` (not in the thesis or paper).
- `MPUGenerator.shared_basis_chain` and `cayley_chain` (also on
  `UniformMPUGenerator`, where one site is reused): site-dependent
  shared-basis counter automata. The reading basis, the vertex unitaries, the
  maps and the bond dimensions may change from site to site; the chain is
  exactly unitary for all of them (the bases of different sites are not
  coupled). `blocking="auto"` is supported. Four new entries of `FAMILIES`:
  `shared_basis_ring`, `shared_basis_general`, `cayley_S3`,
  `cayley_S3_general` (not in the thesis or paper).
- `UniformMPUGenerator.shared_basis_automaton` and `cayley_automaton`: Mode I
  counter automata in which every vertex reads the site in one common basis
  (lazy rings, Cayley graphs such as $S_3$ with two letters, final phases).
  Unitary for every reading basis, vertex unitaries, phases and transition
  maps; `group_automaton` and Mode II `cyclic` with `bond_dim == d` are
  special cases (tested). With permutation maps the operator tail does not
  decay towards larger site index, unlike the Haar-basis ring of `branching`
  (numerical; see the README).
- `fusion(skeleton="cyclic")` accepts `bond_dim > d` (roadmap item). The
  chain is exactly unitary (flat-spectrum boundary), commutes with the total
  charge and has operator-Schmidt rank `bond_dim` once each side holds
  `ceil((bond_dim-1)/(d-1))` sites; use `blocking="auto"`. The linear
  independence check of the single-site channels is skipped for
  `bond_dim > d` (it cannot hold), and a `UserWarning` is issued when the
  chain is too short to reach `bond_dim`. Outputs for `bond_dim <= d` are
  unchanged.
- `MPUGenerator.branching(digraph=..., out_degree=..., ranks=...)`: hand-picked
  or random digraphs and rank partitions for Mode I (random out-degree,
  uniform random composition of `d`, explicit `{vertex: [ranks]}`); every
  choice satisfies row resolution. Non-default choices are different
  ensembles on the same certified class.
- `MPUGenerator(coherent_in_n=False)`: an independent family for every chain
  length, seeded with `(seed, n_sites)`; the default keeps one family for all N.
- `build_family(..., coherent_in_n=..., **options)` forwards both.

### Documentation
- Numpydoc docstrings and comments throughout. Corrected statements: Mode II-B
  unitarity is validated numerically only (the intertwiner helpers are
  standalone and not called at generation); the boundary condition (B) is
  checked only for the `cyclic` and `group` skeletons; with a combined
  readout `blocking` must divide both `n_sites` and `n_sites + m`; skeleton
  size constraints; roadmap; AI usage disclosure.

## 0.2.0 (2026-09-27)

Audit against the thesis ("Generating MPUs") and the tensor-first sampling
paper. All generator outputs are bit-for-bit identical to 0.1.0 (same RNG
draw order; checked by the golden fixtures in the TreeMPU repository).

### Fixed
- `verify=True` now also checks exact unitarity of the dense chain when
  d^N ≤ 2^10; the one-sweep residual is documented as a normalization check
  (necessary, not sufficient).
- `skeletons.solve_intertwiner` accepts rectangular injective intertwiners,
  as in the thesis definition (the Lee–Yang σ·σ = e + σ intertwiner is 9 × 5
  and is now solvable); it no longer claims checks it does not perform.

### Added
- `build_family` / `FAMILIES`: the named families of the thesis and paper
  with their production settings.
- `skeletons.mpo_product_tensor`; `lee_yang_skeleton(alpha, beta)` exposes the
  two-parameter unitary boundary family (default unchanged).
- Package exports `skeletons`, `mixing`, `__version__`.
- Tests: negative tests for every algebraic check, Lee–Yang intertwiner and
  boundary family, registry, reproducibility, CNOT witness.

### Removed
- Unused generic boundary solver `_solve_boundary_general` and an
  unreachable branch of the nested-flag boundary sampler.

### Documentation
- Numpydoc docstrings; internal document references replaced by thesis and
  paper labels; scope of each construction stated (e.g. the cyclic skeleton
  is a group convolution only when χ divides d; the single-tensor
  translation-invariant constructions are not part of thesis or paper).

## 0.1.0
Initial release.
