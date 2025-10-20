"""
Tests for matrix module

Test matrix operations including multiplication, inversion, and eigenvalue decomposition.
"""

import pytest
import numpy as np
from numpy.linalg import LinAlgError

from wannsymm.matrix import (
    matrix3x3_transpose,
    matrix3x3_inverse,
    matrix3x3_dot,
    matrix3x3_determinant,
    matrix3x3_copy,
    matrix3x1_copy,
    matrix_multiply,
    matrix_inverse,
    matrix_eigh,
    matrix_eig,
)


class TestMatrix3x3Transpose:
    """Test 3x3 matrix transpose operation."""
    
    def test_transpose_identity(self):
        """Test transpose of identity matrix."""
        m = np.eye(3)
        mt = matrix3x3_transpose(m)
        assert np.allclose(mt, m)
    
    def test_transpose_general(self):
        """Test transpose of general matrix."""
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        mt = matrix3x3_transpose(m)
        # Transpose swaps rows and columns: mt[i,j] = m[j,i]
        assert mt[0, 1] == 4.0  # mt[0,1] = m[1,0]
        assert mt[1, 0] == 2.0  # mt[1,0] = m[0,1]
        assert mt[2, 1] == 6.0  # mt[2,1] = m[1,2]
    
    def test_transpose_double_transpose(self):
        """Test that double transpose returns original."""
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        mtt = matrix3x3_transpose(matrix3x3_transpose(m))
        assert np.allclose(mtt, m)
    
    def test_transpose_wrong_shape(self):
        """Test that wrong shape raises error."""
        m = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises(ValueError, match="Expected 3x3 matrix"):
            matrix3x3_transpose(m)


class TestMatrix3x3Inverse:
    """Test 3x3 matrix inversion."""
    
    def test_inverse_identity(self):
        """Test inverse of identity matrix."""
        m = np.eye(3)
        mi = matrix3x3_inverse(m)
        assert np.allclose(mi, np.eye(3))
    
    def test_inverse_analytic(self):
        """Test inverse with analytically known result."""
        # Matrix with determinant 2
        m = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
        mi = matrix3x3_inverse(m)
        expected = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=float)
        assert np.allclose(mi, expected)
    
    def test_inverse_multiply_gives_identity(self):
        """Test that M * M^-1 = I."""
        # Use a non-singular matrix with known determinant
        m = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
        mi = matrix3x3_inverse(m)
        product = matrix3x3_dot(m, mi)
        assert np.allclose(product, np.eye(3), atol=1e-10)
    
    def test_inverse_singular_matrix(self):
        """Test that singular matrix raises error."""
        m = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=float)
        with pytest.raises(LinAlgError):
            matrix3x3_inverse(m)
    
    def test_inverse_wrong_shape(self):
        """Test that wrong shape raises error."""
        m = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises(ValueError, match="Expected 3x3 matrix"):
            matrix3x3_inverse(m)


class TestMatrix3x3Dot:
    """Test 3x3 matrix multiplication."""
    
    def test_dot_identity(self):
        """Test multiplication with identity."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        i = np.eye(3)
        c = matrix3x3_dot(a, i)
        assert np.allclose(c, a)
    
    def test_dot_analytic(self):
        """Test multiplication with known result."""
        a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        b = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
        c = matrix3x3_dot(a, b)
        expected = np.array([[2, 0, 0], [0, 6, 0], [0, 0, 12]], dtype=float)
        assert np.allclose(c, expected)
    
    def test_dot_non_commutative(self):
        """Test that matrix multiplication is not commutative."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
        ab = matrix3x3_dot(a, b)
        ba = matrix3x3_dot(b, a)
        assert not np.allclose(ab, ba)
    
    def test_dot_associative(self):
        """Test associativity: (AB)C = A(BC)."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
        c = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)
        
        ab_c = matrix3x3_dot(matrix3x3_dot(a, b), c)
        a_bc = matrix3x3_dot(a, matrix3x3_dot(b, c))
        assert np.allclose(ab_c, a_bc)
    
    def test_dot_wrong_shape(self):
        """Test that wrong shape raises error."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.eye(3)
        with pytest.raises(ValueError, match="Expected 3x3 matrix"):
            matrix3x3_dot(a, b)


class TestMatrix3x3Determinant:
    """Test 3x3 matrix determinant."""
    
    def test_determinant_identity(self):
        """Test determinant of identity matrix."""
        m = np.eye(3)
        det = matrix3x3_determinant(m)
        assert np.isclose(det, 1.0)
    
    def test_determinant_zero(self):
        """Test determinant of singular matrix."""
        m = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=float)
        det = matrix3x3_determinant(m)
        assert np.isclose(det, 0.0, atol=1e-10)
    
    def test_determinant_analytic(self):
        """Test determinant with known result."""
        # Determinant = 2
        m = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
        det = matrix3x3_determinant(m)
        assert np.isclose(det, 2.0)
    
    def test_determinant_negative(self):
        """Test determinant with negative result (reflection)."""
        # Reflection matrix has determinant -1
        m = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        det = matrix3x3_determinant(m)
        assert np.isclose(det, -1.0)
    
    def test_determinant_general(self):
        """Test determinant of general matrix."""
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        det = matrix3x3_determinant(m)
        # Calculate manually: 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7)
        # = 1*(50-48) - 2*(40-42) + 3*(32-35)
        # = 1*2 - 2*(-2) + 3*(-3)
        # = 2 + 4 - 9 = -3
        assert np.isclose(det, -3.0)
    
    def test_determinant_wrong_shape(self):
        """Test that wrong shape raises error."""
        m = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises(ValueError, match="Expected 3x3 matrix"):
            matrix3x3_determinant(m)


class TestMatrix3x3Copy:
    """Test 3x3 matrix and vector copy functions."""
    
    def test_copy_matrix(self):
        """Test matrix copy."""
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        mc = matrix3x3_copy(m)
        assert np.allclose(mc, m)
        assert mc is not m  # Different objects
        # Modify copy, original should not change
        mc[0, 0] = 99
        assert m[0, 0] == 1.0
    
    def test_copy_vector(self):
        """Test vector copy."""
        v = np.array([1.0, 2.0, 3.0])
        vc = matrix3x1_copy(v)
        assert np.allclose(vc, v)
        assert vc is not v  # Different objects
        # Modify copy, original should not change
        vc[0] = 99
        assert v[0] == 1.0
    
    def test_copy_vector_wrong_size(self):
        """Test that wrong size raises error."""
        v = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Expected vector of length 3"):
            matrix3x1_copy(v)


class TestMatrixMultiply:
    """Test general matrix multiplication."""
    
    def test_multiply_real_matrices(self):
        """Test multiplication of real matrices."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[5, 6], [7, 8]], dtype=float)
        c = matrix_multiply(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=float)
        assert np.allclose(c, expected)
    
    def test_multiply_complex_matrices(self):
        """Test multiplication of complex matrices."""
        a = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
        b = np.array([[5, 6+1j], [7, 8]], dtype=complex)
        c = matrix_multiply(a, b)
        # Verify shape
        assert c.shape == (2, 2)
        # Verify result is complex
        assert c.dtype == np.complex128
    
    def test_multiply_with_conj_transpose(self):
        """Test multiplication with conjugate transpose."""
        a = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
        b = np.array([[5, 6+1j], [7, 8]], dtype=complex)
        
        # a^H @ b
        c1 = matrix_multiply(a, b, conj_a=True)
        # Manual calculation
        a_h = np.conj(a.T)
        c2 = np.matmul(a_h, b)
        assert np.allclose(c1, c2)
    
    def test_multiply_incompatible_dimensions(self):
        """Test that incompatible dimensions raise error."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 2, 3]], dtype=float)
        with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
            matrix_multiply(a, b)


class TestMatrixInverse:
    """Test general matrix inversion."""
    
    def test_inverse_2x2(self):
        """Test inversion of 2x2 matrix."""
        m = np.array([[4, 7], [2, 6]], dtype=float)
        mi = matrix_inverse(m)
        product = np.matmul(m, mi)
        assert np.allclose(product, np.eye(2))
    
    def test_inverse_complex_matrix(self):
        """Test inversion of complex matrix."""
        m = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
        mi = matrix_inverse(m)
        product = np.matmul(m, mi)
        assert np.allclose(product, np.eye(2))
    
    def test_inverse_singular_raises_error(self):
        """Test that singular matrix raises LinAlgError."""
        m = np.array([[1, 2], [2, 4]], dtype=float)
        with pytest.raises(LinAlgError, match="singular"):
            matrix_inverse(m)
    
    def test_inverse_non_square_raises_error(self):
        """Test that non-square matrix raises error."""
        m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="must be square"):
            matrix_inverse(m)


class TestMatrixEigh:
    """Test Hermitian eigenvalue decomposition."""
    
    def test_eigh_real_symmetric(self):
        """Test eigenvalues of real symmetric matrix."""
        # Known eigenvalues: -1 and 3
        m = np.array([[1, 2], [2, 1]], dtype=float)
        evals, evecs = matrix_eigh(m)
        assert np.allclose(sorted(evals), [-1, 3])
        
        # Verify eigenvectors
        for i in range(2):
            # M v = lambda v
            mv = np.dot(m, evecs[:, i])
            lv = evals[i] * evecs[:, i]
            assert np.allclose(mv, lv)
    
    def test_eigh_complex_hermitian(self):
        """Test eigenvalues of complex Hermitian matrix."""
        m = np.array([[1, 1-1j], [1+1j, 2]], dtype=complex)
        evals, evecs = matrix_eigh(m)
        
        # Eigenvalues should be real for Hermitian matrix
        assert np.all(np.isreal(evals))
        
        # Verify eigenvectors
        for i in range(2):
            mv = np.dot(m, evecs[:, i])
            lv = evals[i] * evecs[:, i]
            assert np.allclose(mv, lv)
    
    def test_eigh_diagonal_matrix(self):
        """Test eigenvalues of diagonal matrix."""
        m = np.diag([3.0, 1.0, 4.0])
        evals, evecs = matrix_eigh(m)
        # Eigenvalues should be diagonal elements (sorted)
        assert np.allclose(sorted(evals), [1.0, 3.0, 4.0])
    
    def test_eigh_eigenvalues_only(self):
        """Test computing eigenvalues only."""
        m = np.array([[1, 2], [2, 1]], dtype=float)
        evals = matrix_eigh(m, compute_eigenvectors=False)
        assert np.allclose(sorted(evals), [-1, 3])
    
    def test_eigh_non_square_raises_error(self):
        """Test that non-square matrix raises error."""
        m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="must be square"):
            matrix_eigh(m)
    
    def test_eigh_orthogonal_eigenvectors(self):
        """Test that eigenvectors are orthogonal."""
        m = np.array([[1, 2], [2, 1]], dtype=float)
        evals, evecs = matrix_eigh(m)
        
        # Eigenvectors should be orthonormal
        gram = np.dot(evecs.T.conj(), evecs)
        assert np.allclose(gram, np.eye(2))


class TestMatrixEig:
    """Test general eigenvalue decomposition."""
    
    def test_eig_real_matrix(self):
        """Test eigenvalues of real matrix."""
        # Rotation matrix: eigenvalues are +/- i
        m = np.array([[0, -1], [1, 0]], dtype=float)
        evals, evecs = matrix_eig(m)
        
        # Eigenvalues should have magnitude 1
        assert np.allclose(np.abs(evals), 1.0)
        
        # Verify eigenvectors
        for i in range(2):
            mv = np.dot(m, evecs[:, i])
            lv = evals[i] * evecs[:, i]
            assert np.allclose(mv, lv)
    
    def test_eig_complex_matrix(self):
        """Test eigenvalues of complex matrix."""
        m = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
        evals, evecs = matrix_eig(m)
        
        # Verify eigenvectors
        for i in range(2):
            mv = np.dot(m, evecs[:, i])
            lv = evals[i] * evecs[:, i]
            assert np.allclose(mv, lv)
    
    def test_eig_eigenvalues_only(self):
        """Test computing eigenvalues only."""
        m = np.array([[1, 2], [3, 4]], dtype=float)
        evals = matrix_eig(m, compute_left=False, compute_right=False)
        # Just verify we get 2 eigenvalues
        assert len(evals) == 2
    
    def test_eig_non_square_raises_error(self):
        """Test that non-square matrix raises error."""
        m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="must be square"):
            matrix_eig(m)


class TestAnalyticMatrixCases:
    """Test matrix operations against analytic solutions."""
    
    def test_rotation_matrix_properties(self):
        """Test rotation matrix has determinant 1 and is orthogonal."""
        # 90 degree rotation around z-axis
        theta = np.pi / 2
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        
        # Determinant should be 1
        det = matrix3x3_determinant(rot)
        assert np.isclose(det, 1.0)
        
        # Inverse should equal transpose for orthogonal matrix
        rot_inv = matrix3x3_inverse(rot)
        rot_t = matrix3x3_transpose(rot)
        assert np.allclose(rot_inv, rot_t)
    
    def test_pauli_matrices_properties(self):
        """Test Pauli matrix properties."""
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Each Pauli matrix squared is identity
        for sigma in [sigma_x, sigma_y, sigma_z]:
            sigma_sq = matrix_multiply(sigma, sigma)
            assert np.allclose(sigma_sq, np.eye(2))
            
            # Eigenvalues should be +/- 1
            evals = matrix_eigh(sigma, compute_eigenvectors=False)
            assert np.allclose(sorted(evals), [-1, 1])
    
    def test_hermitian_matrix_real_eigenvalues(self):
        """Test that Hermitian matrix has real eigenvalues."""
        # Create random Hermitian matrix
        n = 5
        m_real = np.random.randn(n, n)
        m_imag = np.random.randn(n, n)
        m = m_real + 1j * m_imag
        m_h = (m + m.T.conj()) / 2  # Make Hermitian
        
        evals = matrix_eigh(m_h, compute_eigenvectors=False)
        # All eigenvalues should be real
        assert np.allclose(evals.imag, 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_nearly_singular_matrix(self):
        """Test matrix that is nearly singular."""
        # Matrix with very small determinant
        m = np.array([[1, 2], [2, 4.00001]], dtype=float)
        # Should still be invertible
        mi = matrix_inverse(m)
        product = np.matmul(m, mi)
        assert np.allclose(product, np.eye(2), atol=1e-5)
    
    def test_very_small_eigenvalues(self):
        """Test matrix with very small eigenvalues."""
        m = np.array([[1e-10, 0], [0, 1]], dtype=float)
        evals, evecs = matrix_eigh(m)
        assert len(evals) == 2
        assert np.min(evals) < 1e-9
    
    def test_large_matrices(self):
        """Test operations on larger matrices."""
        n = 100
        m = np.random.randn(n, n)
        m_h = (m + m.T) / 2  # Make symmetric
        
        # Should be able to compute eigenvalues
        evals = matrix_eigh(m_h, compute_eigenvectors=False)
        assert len(evals) == n
        
    def test_integer_matrices(self):
        """Test that integer matrices are handled correctly."""
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=int)
        # Should convert to float
        det = matrix3x3_determinant(m)
        assert isinstance(det, float)
