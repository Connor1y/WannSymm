"""
Tests for rotate_spinor module

Test spinor rotation operations.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from wannsymm.rotate_spinor import (
    rotate_spinor,
    verify_unitarity,
    verify_su2_determinant,
    from_scipy_rotation,
    to_scipy_rotation,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
)


class TestPauliMatrices:
    """Test Pauli matrix definitions."""

    def test_pauli_x(self):
        """Test Pauli X matrix."""
        expected = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        assert np.allclose(PAULI_X, expected)

    def test_pauli_y(self):
        """Test Pauli Y matrix."""
        expected = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        assert np.allclose(PAULI_Y, expected)

    def test_pauli_z(self):
        """Test Pauli Z matrix."""
        expected = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        assert np.allclose(PAULI_Z, expected)

    def test_pauli_commutation_relations(self):
        """Test Pauli matrix commutation relations [σ_i, σ_j] = 2iε_ijk σ_k."""
        # [σ_x, σ_y] = 2i σ_z
        comm_xy = PAULI_X @ PAULI_Y - PAULI_Y @ PAULI_X
        assert np.allclose(comm_xy, 2j * PAULI_Z)

        # [σ_y, σ_z] = 2i σ_x
        comm_yz = PAULI_Y @ PAULI_Z - PAULI_Z @ PAULI_Y
        assert np.allclose(comm_yz, 2j * PAULI_X)

        # [σ_z, σ_x] = 2i σ_y
        comm_zx = PAULI_Z @ PAULI_X - PAULI_X @ PAULI_Z
        assert np.allclose(comm_zx, 2j * PAULI_Y)

    def test_pauli_anticommutation_relations(self):
        """Test Pauli anticommutation {σ_i, σ_j} = 2δ_ij identity_matrix."""
        identity_matrix = np.eye(2, dtype=np.complex128)

        # {σ_x, σ_x} = 2 identity_matrix
        anticomm_xx = PAULI_X @ PAULI_X + PAULI_X @ PAULI_X
        assert np.allclose(anticomm_xx, 2 * identity_matrix)

        # {σ_x, σ_y} = 0
        anticomm_xy = PAULI_X @ PAULI_Y + PAULI_Y @ PAULI_X
        assert np.allclose(anticomm_xy, np.zeros((2, 2)))

    def test_pauli_squares(self):
        """Test that σ_i² = identity_matrix."""
        identity_matrix = np.eye(2, dtype=np.complex128)
        assert np.allclose(PAULI_X @ PAULI_X, identity_matrix)
        assert np.allclose(PAULI_Y @ PAULI_Y, identity_matrix)
        assert np.allclose(PAULI_Z @ PAULI_Z, identity_matrix)


class TestRotateSpinor:
    """Test rotate_spinor function."""

    def test_zero_angle_is_identity(self):
        """Test that 0° rotation gives identity matrix."""
        rot_matrix = rotate_spinor(np.array([0, 0, 1]), 0.0)
        identity_matrix = np.eye(2, dtype=np.complex128)
        assert np.allclose(rot_matrix, identity_matrix, atol=1e-10)

    def test_180_degree_rotation_around_z(self):
        """Test 180° rotation around z-axis."""
        rot_matrix = rotate_spinor(np.array([0, 0, 1]), np.pi)
        # R(π, ẑ) = exp(-iπ/2 σ_z) = -i σ_z
        expected = -1j * PAULI_Z
        assert np.allclose(rot_matrix, expected, atol=1e-10)

    def test_360_degree_rotation_is_minus_identity(self):
        """Test 360° rotation gives -identity_matrix (SU(2) double cover)."""
        # Test for various axes
        for axis in [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 1]) / np.sqrt(3),
        ]:
            rot_matrix = rotate_spinor(axis, 2 * np.pi)
            minus_identity = -np.eye(2, dtype=np.complex128)
            assert np.allclose(rot_matrix, minus_identity, atol=1e-10), (
                f"360° rotation around {axis} should give -identity_matrix"
            )

    def test_720_degree_rotation_is_identity(self):
        """Test that 720° rotation gives identity_matrix."""
        for axis in [
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]) / np.sqrt(2),
        ]:
            rot_matrix = rotate_spinor(axis, 4 * np.pi)
            identity_matrix = np.eye(2, dtype=np.complex128)
            assert np.allclose(rot_matrix, identity_matrix, atol=1e-10), (
                f"720° rotation around {axis} should give identity_matrix"
            )

    def test_90_degree_rotation_around_z(self):
        """Test 90° rotation around z-axis."""
        rot_matrix = rotate_spinor(np.array([0, 0, 1]), np.pi / 2)
        # R(π/2, ẑ) = exp(-iπ/4 σ_z)
        # = cos(π/4)identity_matrix - i sin(π/4)σ_z
        # = (1/√2)(identity_matrix - i σ_z)
        expected = (1 / np.sqrt(2)) * (np.eye(2) - 1j * PAULI_Z)
        assert np.allclose(rot_matrix, expected, atol=1e-10)

    def test_rotation_around_x_axis(self):
        """Test rotation around x-axis."""
        angle = np.pi / 3
        rot_matrix = rotate_spinor(np.array([1, 0, 0]), angle)
        # R(θ, x̂) = cos(θ/2)identity_matrix - i sin(θ/2)σ_x
        expected = (
            np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * PAULI_X
        )
        assert np.allclose(rot_matrix, expected, atol=1e-10)

    def test_rotation_around_y_axis(self):
        """Test rotation around y-axis."""
        angle = np.pi / 4
        rot_matrix = rotate_spinor(np.array([0, 1, 0]), angle)
        # R(θ, ŷ) = cos(θ/2)identity_matrix - i sin(θ/2)σ_y
        expected = (
            np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * PAULI_Y
        )
        assert np.allclose(rot_matrix, expected, atol=1e-10)

    def test_arbitrary_axis_rotation(self):
        """Test rotation around arbitrary axis."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 6
        rot_matrix = rotate_spinor(axis, angle)

        # Verify it's a valid rotation
        assert verify_unitarity(rot_matrix)
        assert verify_su2_determinant(rot_matrix)

    def test_zero_axis_gives_identity(self):
        """Test that zero axis gives identity matrix."""
        rot_matrix = rotate_spinor(np.array([0, 0, 0]), np.pi)
        identity_matrix = np.eye(2, dtype=np.complex128)
        assert np.allclose(rot_matrix, identity_matrix, atol=1e-10)

    def test_non_unit_axis_is_normalized(self):
        """Test that non-unit axis is automatically normalized."""
        # These should give the same result
        r1 = rotate_spinor(np.array([2, 0, 0]), np.pi / 4)
        r2 = rotate_spinor(np.array([1, 0, 0]), np.pi / 4)
        assert np.allclose(r1, r2, atol=1e-10)

    def test_input_validation(self):
        """Test input validation."""
        # Wrong axis shape
        with pytest.raises(ValueError, match="Expected axis of shape"):
            rotate_spinor(np.array([1, 2]), np.pi)


class TestUnitarity:
    """Test unitarity verification."""

    def test_unitarity_of_rotations(self):
        """Test that all rotation matrices are unitary."""
        angles = [
            0,
            np.pi / 6,
            np.pi / 4,
            np.pi / 3,
            np.pi / 2,
            np.pi,
            2 * np.pi,
        ]
        axes = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, 1, 1]) / np.sqrt(3),
        ]

        for axis in axes:
            for angle in angles:
                rot_matrix = rotate_spinor(axis, angle)
                assert verify_unitarity(rot_matrix), (
                    f"Rotation by {angle} around {axis} is not unitary"
                )

    def test_unitarity_property(self):
        """Test R†R = identity_matrix explicitly."""
        rot_matrix = rotate_spinor(
            np.array([1, 2, 3]) / np.sqrt(14), np.pi / 5
        )
        identity_matrix = np.eye(2, dtype=np.complex128)
        product = rot_matrix @ rot_matrix.conj().T
        assert np.allclose(product, identity_matrix, atol=1e-10)

    def test_non_unitary_matrix(self):
        """Test that non-unitary matrix is detected."""
        mat = np.array([[1, 0], [0, 2]], dtype=np.complex128)
        assert not verify_unitarity(mat)


class TestSU2Determinant:
    """Test SU(2) determinant property."""

    def test_determinant_is_one(self):
        """Test that all rotation matrices have determinant 1."""
        angles = [0, np.pi / 6, np.pi / 2, np.pi, 2 * np.pi]
        axes = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 1]) / np.sqrt(3),
        ]

        for axis in axes:
            for angle in angles:
                rot_matrix = rotate_spinor(axis, angle)
                assert verify_su2_determinant(rot_matrix), (
                    f"Rotation by {angle} around {axis} does not have det=1"
                )

    def test_determinant_calculation(self):
        """Test determinant calculation explicitly."""
        rot_matrix = rotate_spinor(np.array([0, 0, 1]), np.pi / 3)
        det = np.linalg.det(rot_matrix)
        assert np.allclose(np.abs(det), 1.0, atol=1e-10)
        assert np.allclose(det, 1.0, atol=1e-10)

    def test_non_su2_matrix(self):
        """Test that non-SU(2) matrix is detected."""
        mat = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        assert not verify_su2_determinant(mat)


class TestScipyIntegration:
    """Test integration with scipy.spatial.transform.Rotation."""

    def test_from_scipy_rotation_identity(self):
        """Test conversion from scipy identity rotation."""
        scipy_rot = Rotation.identity()
        rot_matrix = from_scipy_rotation(scipy_rot)
        identity_matrix = np.eye(2, dtype=np.complex128)
        assert np.allclose(rot_matrix, identity_matrix, atol=1e-10)

    def test_from_scipy_rotation_90deg_z(self):
        """Test conversion from scipy 90° rotation around z."""
        scipy_rot = Rotation.from_euler("z", 90, degrees=True)
        rot_matrix = from_scipy_rotation(scipy_rot)

        # Should match our rotate_spinor result
        expected = rotate_spinor(np.array([0, 0, 1]), np.pi / 2)
        assert np.allclose(rot_matrix, expected, atol=1e-10)

    def test_from_scipy_rotation_arbitrary(self):
        """Test conversion from arbitrary scipy rotation."""
        scipy_rot = Rotation.from_euler("zyx", [30, 45, 60], degrees=True)
        rot_matrix = from_scipy_rotation(scipy_rot)

        # Verify it's a valid SU(2) matrix
        assert verify_unitarity(rot_matrix)
        assert verify_su2_determinant(rot_matrix)

    def test_to_scipy_rotation_identity(self):
        """Test conversion to scipy from identity."""
        rot_matrix = np.eye(2, dtype=np.complex128)
        scipy_rot = to_scipy_rotation(rot_matrix)

        # Should give identity or near-identity rotation
        rotvec = scipy_rot.as_rotvec()
        assert np.allclose(rotvec, [0, 0, 0], atol=1e-10)

    def test_to_scipy_rotation_90deg_z(self):
        """Test conversion to scipy from 90° z rotation."""
        rot_matrix = rotate_spinor(np.array([0, 0, 1]), np.pi / 2)
        scipy_rot = to_scipy_rotation(rot_matrix)

        # Extract angle around z-axis (may differ by sign due to conventions)
        euler = scipy_rot.as_euler("zyx", degrees=True)
        assert np.allclose(np.abs(euler[0]), 90.0, atol=1e-6)

    def test_round_trip_conversion(self):
        """Test that round-trip conversion preserves rotation (up to sign)."""
        # Start with scipy rotation
        scipy_rot_orig = Rotation.from_euler(
            "xyz", [30, 45, 60], degrees=True
        )

        # Convert to SU(2)
        rot_matrix = from_scipy_rotation(scipy_rot_orig)

        # Convert back to scipy
        scipy_rot_back = to_scipy_rotation(rot_matrix)

        # Compare rotation vectors (may differ by sign due to double cover)
        rotvec_orig = scipy_rot_orig.as_rotvec()
        rotvec_back = scipy_rot_back.as_rotvec()

        # Allow for sign ambiguity
        assert np.allclose(rotvec_orig, rotvec_back, atol=1e-6) or np.allclose(
            rotvec_orig, -rotvec_back, atol=1e-6
        )


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_opposite_rotations_compose_to_identity(self):
        """Test that R(θ) @ R(-θ) = identity_matrix."""
        axis = np.array([1, 2, 3]) / np.sqrt(14)
        angle = np.pi / 7

        r_forward = rotate_spinor(axis, angle)
        r_backward = rotate_spinor(axis, -angle)

        identity_matrix = np.eye(2, dtype=np.complex128)
        product = r_forward @ r_backward
        assert np.allclose(product, identity_matrix, atol=1e-10)

    def test_sequential_rotations_same_axis(self):
        """Test that R(θ₁) @ R(θ₂) = R(θ₁ + θ₂) for same axis."""
        axis = np.array([0, 0, 1])
        angle1 = np.pi / 3
        angle2 = np.pi / 6

        r1 = rotate_spinor(axis, angle1)
        r2 = rotate_spinor(axis, angle2)
        r_combined = rotate_spinor(axis, angle1 + angle2)

        product = r1 @ r2
        assert np.allclose(product, r_combined, atol=1e-10)

    def test_very_small_angles(self):
        """Test behavior with very small angles."""
        axis = np.array([1, 0, 0])
        angle = 1e-8

        rot_matrix = rotate_spinor(axis, angle)

        # For small θ: R ≈ identity_matrix - i(θ/2)n·σ
        expected = np.eye(2) - 1j * (angle / 2) * PAULI_X
        assert np.allclose(rot_matrix, expected, atol=1e-10)
        assert verify_unitarity(rot_matrix)

    def test_large_angles(self):
        """Test behavior with large angles (multiple 2π)."""
        axis = np.array([0, 1, 0])
        angle = 10 * np.pi  # 5 full rotations (SU(2) periods)

        rot_matrix = rotate_spinor(axis, angle)

        # 10π = 5 × 2π should give -identity_matrix
        # (odd number of 2π multiples gives -identity_matrix)
        minus_identity = -np.eye(2, dtype=np.complex128)
        assert np.allclose(rot_matrix, minus_identity, atol=1e-10)


class TestPhysicalProperties:
    """Test physical properties of spin rotations."""

    def test_spin_flip_180_degree(self):
        """Test that 180° rotation flips spin states."""
        # 180° rotation around x should flip |↑⟩ ↔ |↓⟩
        rot_matrix = rotate_spinor(np.array([1, 0, 0]), np.pi)

        spin_up = np.array([1, 0], dtype=np.complex128)
        spin_down = np.array([0, 1], dtype=np.complex128)

        # R|↑⟩ should give ±i|↓⟩ (up to phase)
        rotated_up = rot_matrix @ spin_up
        assert np.allclose(np.abs(rotated_up[0]), 0, atol=1e-10)
        assert np.allclose(np.abs(rotated_up[1]), 1, atol=1e-10)

        # R|↓⟩ should give ±i|↑⟩ (up to phase)
        rotated_down = rot_matrix @ spin_down
        assert np.allclose(np.abs(rotated_down[0]), 1, atol=1e-10)
        assert np.allclose(np.abs(rotated_down[1]), 0, atol=1e-10)

    def test_spin_conservation(self):
        """Test that rotation preserves norm of spinor."""
        rot_matrix = rotate_spinor(
            np.array([1, 1, 1]) / np.sqrt(3), np.pi / 5
        )

        # Create random normalized spinor
        spinor = np.array([0.6 + 0.2j, 0.4 - 0.6j], dtype=np.complex128)
        spinor = spinor / np.linalg.norm(spinor)

        rotated = rot_matrix @ spinor

        # Norm should be preserved
        assert np.allclose(
            np.linalg.norm(spinor), np.linalg.norm(rotated), atol=1e-10
        )
        assert np.allclose(np.linalg.norm(rotated), 1.0, atol=1e-10)
