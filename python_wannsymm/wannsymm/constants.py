"""
Constants module for WannSymm

This module contains all constant definitions used throughout the code.
Translated from: src/constants.h

Translation Status: ✓ COMPLETE
"""

import numpy as np
from typing import Final

# String/array buffer sizes
MAXLEN: Final[int] = 512
"""Maximum length for string buffers."""

MEDIUMLEN: Final[int] = 256
"""Medium length for string buffers."""

SHORTLEN: Final[int] = 128
"""Short length for string buffers."""

# Maximum count
MAXN: Final[int] = 10
"""Maximum count for various operations."""

# Tolerance/epsilon values (decreasing precision thresholds)
eps4: Final[float] = 1e-4
"""Tolerance value: 1×10⁻⁴"""

eps5: Final[float] = 1e-5
"""Tolerance value: 1×10⁻⁵"""

eps6: Final[float] = 1e-6
"""Tolerance value: 1×10⁻⁶"""

eps7: Final[float] = 1e-7
"""Tolerance value: 1×10⁻⁷"""

eps8: Final[float] = 1e-8
"""Tolerance value: 1×10⁻⁸"""

# Mathematical constants
sqrt2: Final[float] = 1.41421356237309504880168872421
"""Square root of 2, high precision value."""

PI: Final[float] = 3.14159265358979323846264338328
"""Pi constant, high precision value."""

# Complex unit
cmplx_i: Final[complex] = 1j
"""Complex unit i (√-1)."""

# Array size limits
MAX_NUM_of_SYMM: Final[int] = 1536
"""Maximum number of symmetry operations."""

MAX_NUM_of_atoms: Final[int] = 1024
"""Maximum number of atoms."""

MAX_L: Final[int] = 3
"""Maximum angular momentum quantum number."""

# Type alias for complex numbers
dcomplex = np.complex128
"""Type alias for double precision complex numbers.
Equivalent to C's 'double complex'."""
