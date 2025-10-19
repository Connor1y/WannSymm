"""
Wannier orbital module for WannSymm

Defines Wannier orbital structures and operations.
Translated from: src/wannorb.h and src/wannorb.c

Translation Status: ⏳ NOT STARTED
"""

# TODO: Translate from src/wannorb.h and src/wannorb.c
#
# Original C files: src/wannorb.h (33 lines) + src/wannorb.c (36 lines)
# Total lines: 69
# Complexity: Low
# Estimated effort: 2-3 hours
#
# Translation notes:
# - Create WannOrb class or dataclass
# - Use Vector class from vector.py
# - Store axis as 3 Vector objects or numpy array
#
# struct __wannorb members:
# - site: vector (orbital site)
# - axis[3]: vector array (x, y, z axes)
# - r: int (main quantum number)
# - l: int (angular momentum)
# - mr: int (cubic harmonic indicator)
# - ms: int (spin quantum number: 0=up, 1=down)
#
# Functions to translate:
# - init_wannorb: Initialize orbital with parameters
# - find_index_of_wannorb: Find orbital in array by properties
