"""
Wannier data module for WannSymm

Manages Hamiltonian data structures and I/O.
Translated from: src/wanndata.h and src/wanndata.c

Translation Status: ⏳ NOT STARTED
"""

# TODO: Translate from src/wanndata.h and src/wanndata.c
#
# Original C files: src/wanndata.h (59 lines) + src/wanndata.c (284 lines)
# Total lines: 343
# Complexity: Medium
# Estimated effort: 6-8 hours
#
# Translation notes:
# - Create WannData class
# - Use numpy arrays for ham (complex)
# - Read/write Hamiltonian files in same format as C version
# - Implement proper initialization and cleanup
#
# struct __wanndata members:
# - norb: int (number of orbitals)
# - nrpt: int (number of R-points)
# - ham: complex array (Hamiltonian matrix elements)
# - hamflag: int array (enabled flags)
# - rvec: vector array (R-vectors)
# - weight: int array (degeneracy weights)
# - kvec: vector (k-vector, for Hk only)
#
# Functions to translate:
# - init_wanndata: Initialize structure
# - read_ham: Read Hamiltonian from _hr.dat file
# - write_ham: Write Hamiltonian to _hr.dat file
# - write_reduced_ham: Write reduced Hamiltonian
# - finalize_wanndata: Cleanup
# - find_index_of_ham: Find Hamiltonian element index
#
# Additional structures:
# - hamblock: Block Hamiltonian structure
# - ham_R: R-vector block format
# - Related functions for ham_R operations
# - combine_wanndata: Combine spin-up/down Hamiltonians
