"""
Vector operations module for WannSymm

This module provides vector operations, vector structures, and linked list utilities.
Translated from: src/vector.h and src/vector.c

Translation Status: ⏳ NOT STARTED
"""

# TODO: Translate from src/vector.h and src/vector.c
#
# Original C files: src/vector.h (66 lines) + src/vector.c (346 lines)
# Total lines: 412
# Complexity: Low-Medium
# Estimated effort: 4-6 hours
#
# Translation notes:
# - Use numpy arrays for vector representation (np.array([x, y, z]))
# - Create dataclass or class for Vector with x, y, z properties
# - Create Symm dataclass for symmetry operations
# - Implement VecLList class for linked list (or use Python list)
# - All vector operations should work with numpy arrays
#
# Key structures to translate:
# 1. struct __vector: x, y, z components
# 2. struct __symm: rot[3] (3 vectors), trans (1 vector)
# 3. struct __vec_llist: linked list node
#
# Functions to translate:
# - init_vector, equal, equale, distance
# - cross_product, dot_product, volume_product
# - vector_scale, vector_multiply, vector_add, vector_sub
# - translate_match, isenclosed, find_vector
# - vector_rotate, vector_Rotate
# - vector_norm, vector_normalization, vector_round
# - array2vector, kpt_equivalent, vec_comp
# - vec_llist_* functions (init, add, del, free, pop, find)
