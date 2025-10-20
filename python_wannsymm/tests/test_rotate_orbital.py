"""
Tests for rotate_orbital module

Test orbital rotation operations, Wigner D-matrices, and cubic harmonics.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from wannsymm.rotate_orbital import (
    generate_Lmatrix,
    rotate_Ylm,
    generate_C2Ylm,
    rotate_cubic,
    get_rotation_matrix,
)


class TestGenerateLmatrix:
    """Test angular momentum operator generation."""
    
    def test_l0_returns_zeros(self):
        """Test that l=0 (s orbital) has zero angular momentum."""
        Lmat = generate_Lmatrix(0)
        assert Lmat.shape == (3, 1, 1)
        assert_allclose(Lmat, np.zeros((3, 1, 1)))
    
    def test_l1_commutation_relations(self):
        """Test that l=1 operators satisfy [Li, Lj] = i*ε_ijk*Lk."""
        Lmat = generate_Lmatrix(1)
        Lx, Ly, Lz = Lmat[0], Lmat[1], Lmat[2]
        
        # [Lx, Ly] = i*Lz
        comm_xy = Lx @ Ly - Ly @ Lx
        assert_allclose(comm_xy, 1j * Lz, atol=1e-10)
        
        # [Ly, Lz] = i*Lx
        comm_yz = Ly @ Lz - Lz @ Ly
        assert_allclose(comm_yz, 1j * Lx, atol=1e-10)
        
        # [Lz, Lx] = i*Ly
        comm_zx = Lz @ Lx - Lx @ Lz
        assert_allclose(comm_zx, 1j * Ly, atol=1e-10)
    
    def test_l2_shape(self):
        """Test that l=2 (d orbitals) has correct shape."""
        Lmat = generate_Lmatrix(2)
        assert Lmat.shape == (3, 5, 5)
    
    def test_l3_shape(self):
        """Test that l=3 (f orbitals) has correct shape."""
        Lmat = generate_Lmatrix(3)
        assert Lmat.shape == (3, 7, 7)
    
    def test_lz_eigenvalues(self):
        """Test that Lz has correct eigenvalues m = -l, ..., l."""
        for l in range(4):
            Lmat = generate_Lmatrix(l)
            Lz = Lmat[2]
            eigenvalues = np.diag(Lz).real
            expected = np.arange(-l, l + 1, dtype=float)
            assert_allclose(eigenvalues, expected, atol=1e-10)


class TestRotateYlm:
    """Test spherical harmonics rotation."""
    
    def test_identity_rotation(self):
        """Test that zero angle gives identity."""
        for l in range(4):
            rot = rotate_Ylm(l, [0, 0, 1], 0.0)
            N = 2 * l + 1
            assert_allclose(rot, np.eye(N), atol=1e-10)
    
    def test_rotation_is_unitary(self):
        """Test that rotation matrices are unitary (R†R = I)."""
        for l in range(4):
            rot = rotate_Ylm(l, [1, 1, 0], np.pi / 3)
            N = 2 * l + 1
            unity = rot.conj().T @ rot
            assert_allclose(unity, np.eye(N), atol=1e-10)
    
    def test_rotation_determinant_is_one(self):
        """Test that det(R) = 1 for all rotations."""
        for l in range(4):
            rot = rotate_Ylm(l, [1, 0, 0], np.pi / 4)
            det = np.linalg.det(rot)
            assert_allclose(np.abs(det), 1.0, atol=1e-10)
    
    def test_l0_always_identity(self):
        """Test that s orbitals (l=0) are invariant under rotation."""
        rot = rotate_Ylm(0, [1, 1, 1], np.pi / 2)
        assert_allclose(rot, np.eye(1), atol=1e-10)
    
    def test_90deg_rotation_z_axis_l1(self):
        """Test 90° rotation around z-axis for p orbitals."""
        rot = rotate_Ylm(1, [0, 0, 1], np.pi / 2)
        # For l=1, rotation around z by 90° using R = exp(-i*alpha*Lz):
        # |1,-1> -> exp(-i*pi/2*(-1))|1,-1> = exp(i*pi/2)|1,-1> = i|1,-1>
        # |1,0> -> |1,0>
        # |1,1> -> exp(-i*pi/2*1)|1,1> = exp(-i*pi/2)|1,1> = -i|1,1>
        expected_diag = np.array([np.exp(1j * np.pi / 2), 1.0, np.exp(-1j * np.pi / 2)])
        assert_allclose(np.diag(rot), expected_diag, atol=1e-10)
    
    def test_180deg_rotation_z_axis_l1(self):
        """Test 180° rotation around z-axis for p orbitals."""
        rot = rotate_Ylm(1, [0, 0, 1], np.pi)
        # For l=1, rotation around z by 180°:
        # |1,-1> -> -|1,-1>, |1,0> -> |1,0>, |1,1> -> -|1,1>
        expected_diag = np.array([np.exp(-1j * np.pi), 1.0, np.exp(1j * np.pi)])
        assert_allclose(np.diag(rot), expected_diag, atol=1e-10)
    
    def test_axis_normalization(self):
        """Test that axis is properly normalized."""
        # Same rotation with normalized and unnormalized axis
        rot1 = rotate_Ylm(1, [1, 0, 0], np.pi / 4)
        rot2 = rotate_Ylm(1, [2, 0, 0], np.pi / 4)
        assert_allclose(rot1, rot2, atol=1e-10)


class TestGenerateC2Ylm:
    """Test cubic to spherical harmonics transformation."""
    
    def test_returns_correct_shapes(self):
        """Test that transformation matrices have correct shapes."""
        for l in range(4):
            C2Ylm, Ylm2C = generate_C2Ylm(l)
            N = 2 * l + 1
            assert C2Ylm.shape == (N, N)
            assert Ylm2C.shape == (N, N)
    
    def test_matrices_are_inverses(self):
        """Test that C2Ylm and Ylm2C are inverses."""
        for l in range(4):
            C2Ylm, Ylm2C = generate_C2Ylm(l)
            N = 2 * l + 1
            product1 = C2Ylm @ Ylm2C
            product2 = Ylm2C @ C2Ylm
            assert_allclose(product1, np.eye(N), atol=1e-10)
            assert_allclose(product2, np.eye(N), atol=1e-10)
    
    def test_l0_identity(self):
        """Test that l=0 transformation is identity."""
        C2Ylm, Ylm2C = generate_C2Ylm(0)
        assert_allclose(C2Ylm, np.eye(1), atol=1e-10)
        assert_allclose(Ylm2C, np.eye(1), atol=1e-10)
    
    def test_l1_structure(self):
        """Test that l=1 has expected structure (pz, px, py order)."""
        C2Ylm, Ylm2C = generate_C2Ylm(1)
        # pz should be |1,0> (middle index)
        assert_allclose(abs(Ylm2C[0, 1]), 1.0, atol=1e-10)
    
    def test_invalid_l_raises_error(self):
        """Test that invalid l values raise an error."""
        with pytest.raises(ValueError, match="not supported"):
            generate_C2Ylm(4)


class TestRotateCubic:
    """Test cubic harmonics rotation."""
    
    def test_identity_rotation(self):
        """Test that zero angle gives identity."""
        for l in range(4):
            rot = rotate_cubic(l, [0, 0, 1], 0.0)
            N = 2 * l + 1
            assert_allclose(rot, np.eye(N), atol=1e-10)
    
    def test_rotation_is_unitary(self):
        """Test that rotation matrices are unitary."""
        for l in range(4):
            rot = rotate_cubic(l, [1, 1, 0], np.pi / 3)
            N = 2 * l + 1
            unity = rot.conj().T @ rot
            assert_allclose(unity, np.eye(N), atol=1e-10)
    
    def test_rotation_determinant_is_one(self):
        """Test that det(R) = 1."""
        for l in range(4):
            rot = rotate_cubic(l, [1, 0, 0], np.pi / 4)
            det = np.linalg.det(rot)
            assert_allclose(np.abs(det), 1.0, atol=1e-10)
    
    def test_l0_always_identity(self):
        """Test that s orbitals are rotation invariant."""
        rot = rotate_cubic(0, [1, 1, 1], np.pi / 2)
        assert_allclose(rot, np.eye(1), atol=1e-10)
    
    def test_90deg_rotation_x_axis_l1(self):
        """Test 90° rotation around x-axis for p orbitals."""
        # Using quantum mechanics convention R = exp(-i*alpha*n·L)
        rot = rotate_cubic(1, [1, 0, 0], np.pi / 2)
        # px stays, pz -> py, py -> -pz
        expected = np.array([
            [0, 0, -1],  # pz' = -py
            [0, 1, 0],   # px' = px
            [1, 0, 0]    # py' = pz
        ], dtype=complex)
        assert_allclose(rot, expected, atol=1e-10)
    
    def test_90deg_rotation_y_axis_l1(self):
        """Test 90° rotation around y-axis for p orbitals."""
        # Using quantum mechanics convention R = exp(-i*alpha*n·L)
        rot = rotate_cubic(1, [0, 1, 0], np.pi / 2)
        # py stays, pz -> -px, px -> pz
        expected = np.array([
            [0, -1, 0],  # pz' = -px
            [1, 0, 0],   # px' = pz
            [0, 0, 1]    # py' = py
        ], dtype=complex)
        assert_allclose(rot, expected, atol=1e-10)
    
    def test_90deg_rotation_z_axis_l1(self):
        """Test 90° rotation around z-axis for p orbitals."""
        # Using quantum mechanics convention R = exp(-i*alpha*n·L)
        rot = rotate_cubic(1, [0, 0, 1], np.pi / 2)
        # pz stays, px -> py, py -> -px
        expected = np.array([
            [1, 0, 0],   # pz' = pz
            [0, 0, 1],   # px' = py
            [0, -1, 0]   # py' = -px
        ], dtype=complex)
        assert_allclose(rot, expected, atol=1e-10)
    
    def test_180deg_rotation_x_axis_l1(self):
        """Test 180° rotation around x-axis for p orbitals."""
        # 180° rotation has same effect regardless of sign convention
        rot = rotate_cubic(1, [1, 0, 0], np.pi)
        # px stays, py -> -py, pz -> -pz
        expected = np.array([
            [-1, 0, 0],  # pz' = -pz
            [0, 1, 0],   # px' = px
            [0, 0, -1]   # py' = -py
        ], dtype=complex)
        assert_allclose(rot, expected, atol=1e-10)
    
    def test_composition_of_rotations(self):
        """Test that composing rotations works correctly."""
        # Two 90° rotations = one 180° rotation (in same direction)
        rot_90 = rotate_cubic(1, [0, 0, 1], np.pi / 2)
        rot_180 = rotate_cubic(1, [0, 0, 1], np.pi)
        rot_composed = rot_90 @ rot_90
        assert_allclose(rot_composed, rot_180, atol=1e-10)


class TestGetRotationMatrix:
    """Test the public interface with caching."""
    
    def test_with_cache(self):
        """Test that caching works."""
        rot1 = get_rotation_matrix(1, [1, 0, 0], np.pi / 2, use_cache=True)
        rot2 = get_rotation_matrix(1, [1, 0, 0], np.pi / 2, use_cache=True)
        # Should get same result
        assert_allclose(rot1, rot2, atol=1e-10)
    
    def test_without_cache(self):
        """Test that non-cached version works."""
        rot1 = get_rotation_matrix(1, [1, 0, 0], np.pi / 2, use_cache=False)
        rot2 = get_rotation_matrix(1, [1, 0, 0], np.pi / 2, use_cache=False)
        # Should get same result
        assert_allclose(rot1, rot2, atol=1e-10)
    
    def test_cached_and_uncached_agree(self):
        """Test that cached and uncached versions agree."""
        rot_cached = get_rotation_matrix(2, [1, 1, 1], np.pi / 3, use_cache=True)
        rot_uncached = get_rotation_matrix(2, [1, 1, 1], np.pi / 3, use_cache=False)
        assert_allclose(rot_cached, rot_uncached, atol=1e-10)


class TestOrthogonality:
    """Test that rotation matrices preserve orthogonality."""
    
    def test_rotation_preserves_norm(self):
        """Test that ||Rv|| = ||v|| for all vectors."""
        for l in range(1, 4):
            N = 2 * l + 1
            rot = rotate_cubic(l, [1, 2, 3], 0.7)
            
            # Test with random vector
            v = np.random.randn(N) + 1j * np.random.randn(N)
            v_rot = rot @ v
            
            assert_allclose(np.linalg.norm(v_rot), np.linalg.norm(v), atol=1e-10)
    
    def test_rotation_preserves_inner_product(self):
        """Test that <Ru, Rv> = <u, v>."""
        for l in range(1, 4):
            N = 2 * l + 1
            rot = rotate_cubic(l, [1, 1, 1], 1.2)
            
            # Test with random vectors
            u = np.random.randn(N) + 1j * np.random.randn(N)
            v = np.random.randn(N) + 1j * np.random.randn(N)
            
            u_rot = rot @ u
            v_rot = rot @ v
            
            inner_original = np.vdot(u, v)
            inner_rotated = np.vdot(u_rot, v_rot)
            
            assert_allclose(inner_rotated, inner_original, atol=1e-10)


class TestSpecialRotations:
    """Test specific rotation cases from the C code."""
    
    def test_360deg_rotation_is_identity(self):
        """Test that full rotation returns to identity."""
        for l in range(4):
            rot = rotate_cubic(l, [1, 1, 1], 2 * np.pi)
            N = 2 * l + 1
            # Note: for half-integer spin, 360° gives -I, but for integer l it's I
            assert_allclose(rot, np.eye(N), atol=1e-10)
    
    def test_inverse_rotation(self):
        """Test that R(-θ) = R(θ)^-1."""
        for l in range(1, 4):
            angle = np.pi / 5
            axis = [1, 2, 3]
            
            rot_forward = rotate_cubic(l, axis, angle)
            rot_backward = rotate_cubic(l, axis, -angle)
            
            N = 2 * l + 1
            product = rot_forward @ rot_backward
            assert_allclose(product, np.eye(N), atol=1e-10)
