"""
Basis rotation module for WannSymm

Combines orbital and spinor rotations for complete basis transformation.
Translated from: src/rotate_basis.h and src/rotate_basis.c

Translation Status: ⏳ NOT STARTED
"""

# TODO: Translate from src/rotate_basis.h and src/rotate_basis.c
#
# Original C files: src/rotate_basis.h (16 lines) + src/rotate_basis.c (140 lines)
# Total lines: 156
# Complexity: Medium
# Estimated effort: 6-8 hours
#
# Translation notes:
# - Combines rotate_orbital and rotate_spinor
# - Handles both non-SOC and SOC cases
# - Constructs full unitary transformation matrix
# - Applies to Wannier orbital basis
#
# Key concepts:
# 1. Non-SOC case: Only orbital rotation needed
# 2. SOC case: Tensor product of orbital and spin rotations
# 3. Unitary matrix: Preserves norm
# 4. Basis transformation: ψ' = U ψ
#
# Basis structure:
# Non-SOC: |orbital⟩
# SOC: |orbital, spin⟩ = |orbital⟩ ⊗ |spin⟩
#
# Functions to translate:
# - rotate_basis: Main function to rotate full basis
# - construct_rotation_matrix: Build rotation matrix
#   - For non-SOC: Just orbital rotation
#   - For SOC: Kronecker product of orbital and spin
# - apply_basis_rotation: Apply rotation to basis set
# - Helper functions for basis manipulation
#
# Dependencies:
# - rotate_orbital for orbital part
# - rotate_spinor for spin part
# - numpy.kron for tensor products in SOC case
