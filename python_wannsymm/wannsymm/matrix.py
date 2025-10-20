"""
Matrix operations module for WannSymm

Provides matrix utilities using numpy/scipy.
Translated from: src/matrix.h and src/matrix.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError


# Type aliases for clarity
RealMatrix = npt.NDArray[np.float64]
ComplexMatrix = npt.NDArray[np.complex128]
Matrix = Union[RealMatrix, ComplexMatrix]


def matrix3x3_transpose(matrix_in: RealMatrix) -> RealMatrix:
    """
    Transpose a 3x3 matrix.
    
    Equivalent to C function: void matrix3x3_transpose(double out[3][3], double in[3][3])
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input 3x3 matrix of shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Transposed 3x3 matrix of shape (3, 3)
    
    Examples
    --------
    >>> m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> mt = matrix3x3_transpose(m)
    >>> mt[0, 1]
    4.0
    """
    matrix_in = np.asarray(matrix_in, dtype=np.float64)
    if matrix_in.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {matrix_in.shape}")
    return matrix_in.T.copy()


def matrix3x3_inverse(matrix_in: RealMatrix) -> RealMatrix:
    """
    Invert a 3x3 matrix.
    
    Equivalent to C function: void matrix3x3_inverse(double out[3][3], double in[3][3])
    Uses LAPACK dgetrf/dgetri internally in C, replaced with numpy.linalg.inv.
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input 3x3 matrix of shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Inverted 3x3 matrix of shape (3, 3)
    
    Raises
    ------
    LinAlgError
        If matrix is singular and cannot be inverted
    
    Examples
    --------
    >>> m = np.eye(3)
    >>> mi = matrix3x3_inverse(m)
    >>> np.allclose(mi, np.eye(3))
    True
    """
    matrix_in = np.asarray(matrix_in, dtype=np.float64)
    if matrix_in.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {matrix_in.shape}")
    
    try:
        return np.linalg.inv(matrix_in)
    except LinAlgError as e:
        raise LinAlgError(f"Matrix is singular and cannot be inverted: {e}") from e


def matrix3x3_dot(a: RealMatrix, b: RealMatrix) -> RealMatrix:
    """
    Multiply two 3x3 matrices (matrix product).
    
    Equivalent to C function: void matrix3x3_dot(double out[3][3], double a[3][3], double b[3][3])
    Uses cblas_dgemm in C, replaced with numpy matrix multiplication.
    
    Parameters
    ----------
    a : np.ndarray
        First 3x3 matrix of shape (3, 3)
    b : np.ndarray
        Second 3x3 matrix of shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Matrix product a @ b of shape (3, 3)
    
    Examples
    --------
    >>> a = np.eye(3)
    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> c = matrix3x3_dot(a, b)
    >>> np.allclose(c, b)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix for a, got shape {a.shape}")
    if b.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix for b, got shape {b.shape}")
    
    return np.matmul(a, b)


def matrix3x3_determinant(m: RealMatrix) -> float:
    """
    Calculate determinant of a 3x3 matrix.
    
    Equivalent to C function: double matrix3x3_determinant(double m[3][3])
    
    Parameters
    ----------
    m : np.ndarray
        Input 3x3 matrix of shape (3, 3)
    
    Returns
    -------
    float
        Determinant of the matrix
    
    Examples
    --------
    >>> m = np.eye(3)
    >>> matrix3x3_determinant(m)
    1.0
    """
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {m.shape}")
    
    return float(np.linalg.det(m))


def matrix3x3_copy(matrix_in: RealMatrix) -> RealMatrix:
    """
    Copy a 3x3 matrix.
    
    Equivalent to C function: void matrix3x3_copy(double out[3][3], double in[3][3])
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input 3x3 matrix of shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Copy of input matrix
    
    Examples
    --------
    >>> m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> mc = matrix3x3_copy(m)
    >>> np.allclose(mc, m)
    True
    >>> mc is m
    False
    """
    matrix_in = np.asarray(matrix_in, dtype=np.float64)
    if matrix_in.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {matrix_in.shape}")
    
    return matrix_in.copy()


def matrix3x1_copy(vector_in: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Copy a 3x1 vector (array of length 3).
    
    Equivalent to C function: void matrix3x1_copy(double out[3], double in[3])
    
    Parameters
    ----------
    vector_in : np.ndarray
        Input vector of length 3
    
    Returns
    -------
    np.ndarray
        Copy of input vector
    
    Examples
    --------
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> vc = matrix3x1_copy(v)
    >>> np.allclose(vc, v)
    True
    >>> vc is v
    False
    """
    vector_in = np.asarray(vector_in, dtype=np.float64).flatten()
    if vector_in.shape != (3,):
        raise ValueError(f"Expected vector of length 3, got shape {vector_in.shape}")
    
    return vector_in.copy()


# ============================================================================
# General Matrix Operations (for arbitrary size matrices)
# ============================================================================

def matrix_multiply(
    a: Matrix,
    b: Matrix,
    conj_a: bool = False,
    conj_b: bool = False
) -> Matrix:
    """
    Multiply two matrices with optional conjugate transpose.
    
    Supports both real and complex matrices. Wraps numpy matrix multiplication
    with options for conjugate transpose, similar to BLAS zgemm/dgemm.
    
    Parameters
    ----------
    a : np.ndarray
        First matrix of shape (m, k)
    b : np.ndarray
        Second matrix of shape (k, n) or (n, k) if conj_b=True
    conj_a : bool, optional
        If True, use conjugate transpose of a (default: False)
    conj_b : bool, optional
        If True, use conjugate transpose of b (default: False)
    
    Returns
    -------
    np.ndarray
        Matrix product of shape (m, n)
    
    Raises
    ------
    ValueError
        If matrix dimensions are incompatible
    
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]], dtype=float)
    >>> b = np.array([[5, 6], [7, 8]], dtype=float)
    >>> c = matrix_multiply(a, b)
    >>> c[0, 0]
    19.0
    
    >>> # Complex matrices
    >>> a_c = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
    >>> b_c = np.array([[5, 6+1j], [7, 8]], dtype=complex)
    >>> c_c = matrix_multiply(a_c, b_c)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Apply conjugate transpose if requested
    if conj_a:
        a = np.conj(a.T)
    if conj_b:
        b = np.conj(b.T)
    
    # Check dimension compatibility
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible matrix dimensions: a.shape={a.shape}, b.shape={b.shape}"
        )
    
    return np.matmul(a, b)


def matrix_inverse(matrix_in: Matrix, hermitian: bool = False) -> Matrix:
    """
    Invert a matrix.
    
    Supports both real and complex matrices. For Hermitian matrices,
    can use specialized algorithms for better numerical stability.
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input square matrix
    hermitian : bool, optional
        If True, assume matrix is Hermitian (default: False)
    
    Returns
    -------
    np.ndarray
        Inverse of the input matrix
    
    Raises
    ------
    ValueError
        If matrix is not square
    LinAlgError
        If matrix is singular and cannot be inverted
    
    Examples
    --------
    >>> m = np.array([[4, 7], [2, 6]], dtype=float)
    >>> mi = matrix_inverse(m)
    >>> np.allclose(np.matmul(m, mi), np.eye(2))
    True
    
    >>> # Complex matrix
    >>> m_c = np.array([[1+1j, 2], [3, 4-1j]], dtype=complex)
    >>> mi_c = matrix_inverse(m_c)
    """
    matrix_in = np.asarray(matrix_in)
    
    if matrix_in.shape[0] != matrix_in.shape[1]:
        raise ValueError(
            f"Matrix must be square, got shape {matrix_in.shape}"
        )
    
    try:
        return np.linalg.inv(matrix_in)
    except LinAlgError as e:
        raise LinAlgError(
            f"Matrix is singular and cannot be inverted: {e}"
        ) from e


def matrix_eigh(
    matrix_in: Matrix,
    compute_eigenvectors: bool = True
) -> Union[Tuple[npt.NDArray[np.float64], ComplexMatrix], npt.NDArray[np.float64]]:
    """
    Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix.
    
    For Hermitian (complex) or symmetric (real) matrices, eigenvalues are
    always real. This function wraps numpy.linalg.eigh which is optimized
    for Hermitian matrices.
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input Hermitian or symmetric matrix
    compute_eigenvectors : bool, optional
        If True, compute eigenvectors (default: True)
    
    Returns
    -------
    eigenvalues : np.ndarray
        Real eigenvalues in ascending order
    eigenvectors : np.ndarray, optional
        Eigenvectors as columns (only if compute_eigenvectors=True)
    
    Raises
    ------
    ValueError
        If matrix is not square
    
    Examples
    --------
    >>> # Real symmetric matrix
    >>> m = np.array([[1, 2], [2, 1]], dtype=float)
    >>> evals, evecs = matrix_eigh(m)
    >>> # Eigenvalues should be [-1, 3]
    >>> np.allclose(evals, [-1, 3])
    True
    
    >>> # Complex Hermitian matrix
    >>> m_c = np.array([[1, 1-1j], [1+1j, 2]], dtype=complex)
    >>> evals_c, evecs_c = matrix_eigh(m_c)
    """
    matrix_in = np.asarray(matrix_in)
    
    if matrix_in.shape[0] != matrix_in.shape[1]:
        raise ValueError(
            f"Matrix must be square, got shape {matrix_in.shape}"
        )
    
    if compute_eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix_in)
        return eigenvalues, eigenvectors
    else:
        eigenvalues = np.linalg.eigvalsh(matrix_in)
        return eigenvalues


def matrix_eig(
    matrix_in: Matrix,
    compute_left: bool = False,
    compute_right: bool = True
) -> Union[
    Tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix],
    Tuple[ComplexMatrix, ComplexMatrix],
    ComplexMatrix
]:
    """
    Compute eigenvalues and eigenvectors of a general matrix.
    
    For non-Hermitian matrices. This wraps numpy.linalg.eig.
    Similar to LAPACK zgeev/dgeev.
    
    Parameters
    ----------
    matrix_in : np.ndarray
        Input square matrix
    compute_left : bool, optional
        If True, compute left eigenvectors (default: False)
    compute_right : bool, optional
        If True, compute right eigenvectors (default: True)
    
    Returns
    -------
    eigenvalues : np.ndarray
        Complex eigenvalues
    left_eigenvectors : np.ndarray, optional
        Left eigenvectors as columns (only if compute_left=True)
    right_eigenvectors : np.ndarray, optional
        Right eigenvectors as columns (only if compute_right=True)
    
    Raises
    ------
    ValueError
        If matrix is not square
    
    Examples
    --------
    >>> m = np.array([[0, -1], [1, 0]], dtype=float)
    >>> evals, evecs = matrix_eig(m)
    >>> # Eigenvalues should be +/- i
    >>> np.allclose(np.abs(evals), 1.0)
    True
    """
    matrix_in = np.asarray(matrix_in)
    
    if matrix_in.shape[0] != matrix_in.shape[1]:
        raise ValueError(
            f"Matrix must be square, got shape {matrix_in.shape}"
        )
    
    if compute_left and compute_right:
        eigenvalues, right_eigenvectors = np.linalg.eig(matrix_in)
        # numpy doesn't directly compute left eigenvectors, 
        # but they can be obtained from the inverse transpose
        left_eigenvalues, left_eigenvectors = np.linalg.eig(matrix_in.T.conj())
        return eigenvalues, left_eigenvectors, right_eigenvectors
    elif compute_right:
        eigenvalues, right_eigenvectors = np.linalg.eig(matrix_in)
        return eigenvalues, right_eigenvectors
    elif compute_left:
        left_eigenvalues, left_eigenvectors = np.linalg.eig(matrix_in.T.conj())
        return left_eigenvalues, left_eigenvectors
    else:
        eigenvalues = np.linalg.eigvals(matrix_in)
        return eigenvalues
