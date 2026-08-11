# MPU_Generation

Certified random generators for Matrix Product Unitaries (MPUs): sample
tensors that are provably unitary for every chain length $N$ under a
single fixed generative rule, rather than approximately unitary or
unitary-on-average.

Two classes, sharing one construction:

- `MPUGenerator` — general (non-uniform), site-resampled MPUs.
- `UniformMPUGenerator(MPUGenerator)` — translationally invariant MPUs
  (single repeating bulk tensor).

Implements Mode I (row resolution / branching), Mode II (fusion closure:
cyclic, nested-flag, matrix-units, and general finite-group skeletons —
abelian or not), Mode II-B (bond-resolved fusion, e.g. Lee-Yang), the
grafted (Theorem 4) composition, and several translationally-invariant-only
mechanisms (self-direction ansatz, resonant, clock, multi-symbol,
invariant-subspace, degenerate-rows), plus mixing combinators
(stack/nest/capped-sum).

Every mechanism's certificate is checked directly (the defining algebraic
condition — row resolution, fusion closure, the Intertwiner condition, or
the self-direction closed form — not inferred from unitarity alone), and
every construction is additionally verified end-to-end via dense-matrix
contraction before being trusted.

## Install

```bash
pip install git+https://github.com/Mrcocvs02/MPU_Generation.git
```

Depends on [`mpu_decomposition`](https://github.com/Mrcocvs02/MPU_Decomposition)
(installed automatically) for boundary-operator sweeps, canonicalization,
and the minimal-bond-dimension check; this package only ever imports
*from* it, never the reverse.

## Reference

The theory implemented here is derived and proved in the author's
undergraduate thesis (Marco Vassallo, Alma Mater Studiorum — Università
di Bologna, expected graduation September 2026), which will be
downloadable from [AMS Laurea](https://amslaurea.unibo.it/) by searching
"Marco Vassallo" once published, and builds on the circuit-synthesis
construction of:

> G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum Circuits for
> Matrix-Product Unitaries", *Phys. Rev. Lett.* **135**, 260602 (2025).
> [arXiv:2508.08160](https://arxiv.org/abs/2508.08160),
> [doi:10.1103/yshb-hmml](https://doi.org/10.1103/yshb-hmml).

## Status

Pre-alpha, under active development. Used as a submodule of
[`MPU_Decomposition`](https://github.com/Mrcocvs02/MPU_Decomposition).

## License

MIT
