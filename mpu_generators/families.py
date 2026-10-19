# mpu_generators/families.py
r"""Named MPU families used in the thesis and in the tensor-first sampling paper.

:data:`FAMILIES` maps each family name to a short description, and
:func:`build_family` builds one instance with exactly the generator
settings used for the numerical results of the thesis (Chapter "Numerical
results"), so that e.g. ``build_family("branching_uniform", d=2, bond_dim=2,
seed=3)`` reproduces that instance.

========================================  ======  =========================================
name                                      mode    construction
========================================  ======  =========================================
``branching_uniform``                     I       translation-invariant row resolution
``branching_general``                     I       site-dependent row resolution
``grafted``                               I+II    Mode I path grafted into cyclic channels
``fusion_cyclic``                         II      cyclic characters
``fusion_nested_flag``                    II      nested flags (contains MCZ)
``fusion_matrix_units``                   II      identity + matrix units
``fusion_group_Zn``                       II      group convolution, :math:`\mathbb Z_n`
``fusion_group_S3``                       II      group convolution, :math:`S_3` (d = 6)
``fusion_group_Zn_product``               II      group convolution, :math:`\mathbb Z_2^3` (d = 8)
``fusion_bond_resolved_lee_yang``         II-B    Lee-Yang skeleton (d = 5, chi = 14)
========================================  ======  =========================================

The tensor-first sampling paper uses ``branching_uniform`` and the Mode II
fusion families; ``branching_general``, ``grafted`` and the Lee-Yang family
appear in the thesis only.
"""

from __future__ import annotations

import math

import numpy as np

FAMILIES: dict[str, str] = {
    "branching_uniform": "Mode I, translation invariant (UniformMPUGenerator.branching)",
    "branching_general": "Mode I, site dependent (MPUGenerator.branching)",
    "grafted": "Grafted composition (UniformMPUGenerator.grafted)",
    "fusion_cyclic": "Mode II, cyclic skeleton (requires bond_dim <= d)",
    "fusion_nested_flag": "Mode II, nested flags (requires bond_dim <= d)",
    "fusion_matrix_units": "Mode II, matrix units (bond_dim = 1 + m^2, m <= d)",
    "fusion_group_Zn": "Mode II, group convolution Z_n (requires d = bond_dim = n)",
    "fusion_group_S3": "Mode II, group convolution S_3 (d = bond_dim = 6)",
    "fusion_group_Zn_product": "Mode II, group convolution Z_2 x Z_2 x Z_2 (d = bond_dim = 8)",
    "fusion_bond_resolved_lee_yang": "Mode II-B, Lee-Yang skeleton (d = 5, bond_dim = 14)",
}


def build_family(
    name: str,
    d: int = 2,
    bond_dim: int = 2,
    seed: int = 0,
    n_sites: int | None = None,
    coherent_in_n: bool = True,
    **options,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build one instance of a named family.

    Parameters
    ----------
    name : str
        A key of :data:`FAMILIES`.
    d : int
        Local physical dimension (ignored by families with a fixed ``d``).
    bond_dim : int
        Bond dimension. For ``grafted`` the bond dimension used is
        ``2 * max(1, bond_dim // 2)``.
    seed : int
        Generator seed.
    n_sites : int, optional
        Number of physical sites. If omitted, a two-tensor chain is built
        (``n_blocks=2``).
    coherent_in_n : bool
        Passed to the generator (see :class:`~mpu_generators.base.MPUGenerator`).
    **options
        Extra keyword arguments of the underlying generator method, e.g.
        ``digraph``, ``out_degree``, ``ranks`` for the Mode I families.

    Returns
    -------
    tensors, l_vec, r_vec
        Site tensors ``A[p_in, p_out, bond_L, bond_R]`` and boundaries.

    Raises
    ------
    KeyError
        Unknown family name.
    ValueError
        The ``(d, bond_dim)`` combination is not valid for this family.

    Notes
    -----
    Mode I families use ``blocking="auto", max_blocking=2`` (the smallest
    block size, up to 2, whose chain has full-rank boundary operators at
    every cut, ``TreeMPU.utils.satisfies_assumption_1``), all others
    ``blocking=1``, as in the thesis production runs.
    """
    from .base import MPUGenerator
    from .uniform import UniformMPUGenerator

    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}")
    size = {"n_sites": n_sites} if n_sites else {"n_blocks": 2}
    U = UniformMPUGenerator(seed=seed, coherent_in_n=coherent_in_n)
    size.update(options)

    if name == "branching_uniform":
        return U.branching(d=d, bond_dim=bond_dim, blocking="auto", max_blocking=2, **size)
    if name == "branching_general":
        return MPUGenerator(seed=seed, coherent_in_n=coherent_in_n).branching(
            d=d, bond_dim=bond_dim, blocking="auto", max_blocking=2, **size
        )
    if name == "grafted":
        n_channels = max(1, bond_dim // 2)
        return U.grafted(d=d, bond_dim=2 * n_channels, blocking=1, **size)
    if name in ("fusion_cyclic", "fusion_nested_flag"):
        skeleton = "cyclic" if name == "fusion_cyclic" else "nested_flag"
        return U.fusion(d=d, bond_dim=bond_dim, blocking=1, skeleton=skeleton, **size)
    if name == "fusion_matrix_units":
        m = math.isqrt(bond_dim - 1) if bond_dim >= 2 else -1
        if bond_dim < 2 or 1 + m * m != bond_dim or m > d:
            raise ValueError("fusion_matrix_units needs bond_dim = 1 + m^2 with 1 <= m <= d")
        return U.fusion(d=d, bond_dim=bond_dim, blocking=1, skeleton="matrix_units", **size)
    if name == "fusion_group_Zn":
        if d != bond_dim:
            raise ValueError("fusion_group_Zn needs d == bond_dim")
        return U.fusion(
            d=d, bond_dim=d, blocking=1, skeleton="group", group="Zn", group_n=d, **size
        )
    if name == "fusion_group_S3":
        return U.fusion(d=6, bond_dim=6, blocking=1, skeleton="group", group="S3", **size)
    if name == "fusion_group_Zn_product":
        return U.fusion(
            d=8,
            bond_dim=8,
            blocking=1,
            skeleton="group",
            group="Zn_product",
            group_n=[2, 2, 2],
            **size,
        )
    # fusion_bond_resolved_lee_yang
    from .skeletons import lee_yang_skeleton

    return U.fusion_bond_resolved(lee_yang_skeleton(), blocking=1, **size)
