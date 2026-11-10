# mpu_generators/__init__.py
"""Certified random generators of exact open-boundary Matrix Product Unitaries.

Main entry points
-----------------
:class:`MPUGenerator`, :class:`UniformMPUGenerator`
    Mode I (row resolution), Mode II (fusion closure), Mode II-B
    (bond-resolved fusion) and grafted constructions, and the shared-basis
    counter automata (``shared_basis_chain``, ``cayley_chain``) and Mode I with
    several starts (``multi_start``).
:func:`generate_random_mpu`
    Functional interface to the generators.
:func:`build_family`, :data:`FAMILIES`
    Named families used in the thesis and the tensor-first sampling paper,
    plus the shared-basis and multi-start families.
:mod:`mpu_generators.skeletons`, :mod:`mpu_generators.mixing`
    The skeleton of every mode (``MPUGenerator.sample`` is the common
    pipeline) and ways to combine certified chains.
"""

from . import mixing, skeletons
from .base import MPUGenerator, generate_random_mpu
from .families import FAMILIES, build_family
from .uniform import UniformMPUGenerator

__version__ = "0.3.0"

__all__ = [
    "FAMILIES",
    "MPUGenerator",
    "UniformMPUGenerator",
    "__version__",
    "build_family",
    "generate_random_mpu",
    "mixing",
    "skeletons",
]
