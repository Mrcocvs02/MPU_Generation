# mpu_generators/__init__.py
"""Certified random generators of exact open-boundary Matrix Product Unitaries.

Main entry points
-----------------
:class:`MPUGenerator`, :class:`UniformMPUGenerator`
    Mode I (row resolution), Mode II (fusion closure), Mode II-B
    (bond-resolved fusion) and grafted constructions.
:func:`generate_random_mpu`
    Functional interface to the generators.
:func:`build_family`, :data:`FAMILIES`
    Named families used in the thesis and the tensor-first sampling paper.
:mod:`mpu_generators.skeletons`, :mod:`mpu_generators.mixing`
    Mode II-B skeletons and ways to combine certified chains.
"""

from . import mixing, skeletons
from .base import MPUGenerator, generate_random_mpu
from .families import FAMILIES, build_family
from .uniform import UniformMPUGenerator

__version__ = "0.2.0"

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
