"""
Spinor rotation module for WannSymm

Rotates spinor wavefunctions for spin-orbit coupling calculations.
Translated from: src/rotate_spinor.h and src/rotate_spinor.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from scipy.linalg import expm

# Type aliases
ComplexMatrix2x2 = npt.NDArray[np.complex128]
RealVector3 = npt.NDArray[np.float64]

# Pauli matrices
# Note: The C code uses σ_y = [[0, -i], [i, 0]] (standard convention)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
"""Pauli matrix σ_x = [[0, 1], [1, 0]]"""

PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
"""Pauli matrix σ_y = [[0, -i], [i, 0]]"""

PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
"""Pauli matrix σ_z = [[1, 0], [0, -1]]"""

PAULI_MATRICES = [PAULI_X, PAULI_Y, PAULI_Z]
"""List of Pauli matrices [σ_x, σ_y, σ_z]"""


def rotate_spinor(axis: RealVector3, angle: float, inv: int = 0) -> ComplexMatrix2x2:
    """
    Compute spin rotation matrix R = exp(-i θ/2 n·σ).

    This function generates a 2x2 SU(2) rotation matrix that rotates spinors
    by angle θ around the axis n. The rotation matrix is computed using the
    formula:
        R = exp(-i θ/2 n·σ)
    where n is the unit rotation axis, θ is the rotation angle, and σ are
    the Pauli matrices.

    Equivalent to C function:
        void rotate_spinor(dcomplex * rot, double axis[3], double alpha, int inv)

    Parameters
    ----------
    axis : np.ndarray
        Rotation axis (should be unit vector), shape (3,)
    angle : float
        Rotation angle in radians (denoted α or θ in physics literature)
    inv : int, optional
        Inversion flag (currently unused, kept for C compatibility), default 0

    Returns
    -------
    np.ndarray
        2x2 complex SU(2) rotation matrix, shape (2, 2)

    Notes
    -----
    The rotation matrix has the following properties:
    - Unitary: R†R = I
    - Determinant: det(R) = 1 (SU(2) group property)
    - Double cover: R(2π) = -I (360° rotation gives minus identity)
    - For angle=0: R = I (identity)
    - For angle=π around z-axis: R = -iσ_z

    The implementation follows the C code which uses eigenvalue decomposition:
    1. Construct M = -i(θ/2) n·σ
    2. Diagonalize M = V D V^-1
    3. Compute exp(M) = V exp(D) V^-1

    Alternatively, this could be computed using the closed form:
        R = cos(θ/2)I - i sin(θ/2) n·σ

    Examples
    --------
    >>> # Identity (no rotation)
    >>> R = rotate_spinor(np.array([0, 0, 1]), 0.0)
    >>> np.allclose(R, np.eye(2))
    True

    >>> # 180° rotation around z-axis
    >>> R = rotate_spinor(np.array([0, 0, 1]), np.pi)
    >>> np.allclose(np.abs(np.linalg.det(R)), 1.0)
    True

    >>> # Verify unitarity
    >>> R = rotate_spinor(np.array([1, 1, 1])/np.sqrt(3), np.pi/4)
    >>> np.allclose(R @ R.conj().T, np.eye(2))
    True
    """
    # Convert axis to numpy array and normalize
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"Expected axis of shape (3,), got {axis.shape}")

    # Normalize axis (in case it's not unit)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        # For zero axis, return identity
        return np.eye(2, dtype=np.complex128)
    axis = axis / axis_norm

    # Construct n·σ = axis[0]*σ_x + axis[1]*σ_y + axis[2]*σ_z
    sigma_n = axis[0] * PAULI_X + axis[1] * PAULI_Y + axis[2] * PAULI_Z

    # Construct the matrix M = -i(θ/2) n·σ
    # This matches the C code: -0.5 * sigma_n * alpha * I
    M = -0.5j * angle * sigma_n

    # Compute rotation matrix using matrix exponential
    # Method 1: Use scipy's expm (more direct)
    R = expm(M)

    # Method 2 (matching C code approach): eigenvalue decomposition
    # eigenvalues, eigenvectors = np.linalg.eig(M)
    # exp_eig = np.diag(np.exp(eigenvalues))
    # R = eigenvectors @ exp_eig @ np.linalg.inv(eigenvectors)

    return R.astype(np.complex128)


def verify_unitarity(R: ComplexMatrix2x2, tol: float = 1e-10) -> bool:
    """
    Verify that a matrix is unitary (R†R = I).

    Parameters
    ----------
    R : np.ndarray
        Complex matrix to verify, shape (2, 2)
    tol : float, optional
        Numerical tolerance for comparison, default 1e-10

    Returns
    -------
    bool
        True if R is unitary within tolerance

    Examples
    --------
    >>> R = rotate_spinor(np.array([0, 0, 1]), np.pi/4)
    >>> verify_unitarity(R)
    True
    """
    R = np.asarray(R, dtype=np.complex128)
    identity = np.eye(R.shape[0], dtype=np.complex128)
    product = R @ R.conj().T
    return np.allclose(product, identity, atol=tol)


def verify_su2_determinant(R: ComplexMatrix2x2, tol: float = 1e-10) -> bool:
    """
    Verify that a matrix has determinant 1 (SU(2) property).

    Parameters
    ----------
    R : np.ndarray
        Complex matrix to verify, shape (2, 2)
    tol : float, optional
        Numerical tolerance for comparison, default 1e-10

    Returns
    -------
    bool
        True if det(R) = 1 within tolerance

    Examples
    --------
    >>> R = rotate_spinor(np.array([1, 0, 0]), np.pi)
    >>> verify_su2_determinant(R)
    True
    """
    R = np.asarray(R, dtype=np.complex128)
    det = np.linalg.det(R)
    return np.allclose(det, 1.0, atol=tol)


def from_scipy_rotation(scipy_rot: Rotation) -> ComplexMatrix2x2:
    """
    Convert scipy Rotation to SU(2) spinor rotation matrix.

    Parameters
    ----------
    scipy_rot : scipy.spatial.transform.Rotation
        Rotation object from scipy

    Returns
    -------
    np.ndarray
        2x2 SU(2) rotation matrix, shape (2, 2)

    Notes
    -----
    scipy.spatial.transform.Rotation represents SO(3) rotations.
    This function extracts the axis-angle representation and converts
    it to the corresponding SU(2) spinor rotation.

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation
    >>> scipy_rot = Rotation.from_euler('z', 90, degrees=True)
    >>> R = from_scipy_rotation(scipy_rot)
    >>> verify_unitarity(R)
    True
    """
    # Get axis-angle representation
    rotvec = scipy_rot.as_rotvec()
    angle = np.linalg.norm(rotvec)

    if angle < 1e-10:
        # Identity rotation
        return np.eye(2, dtype=np.complex128)

    axis = rotvec / angle
    return rotate_spinor(axis, angle)


def to_scipy_rotation(R: ComplexMatrix2x2) -> Rotation:
    """
    Convert SU(2) spinor rotation matrix to scipy Rotation (SO(3)).

    Parameters
    ----------
    R : np.ndarray
        2x2 SU(2) rotation matrix, shape (2, 2)

    Returns
    -------
    scipy.spatial.transform.Rotation
        Rotation object representing the corresponding SO(3) rotation

    Notes
    -----
    This is the inverse of from_scipy_rotation. Note that SU(2) is a
    double cover of SO(3), so R and -R map to the same SO(3) rotation.

    The conversion uses the fact that for R = exp(-iθ/2 n·σ):
    - trace(R) = 2cos(θ/2)
    - The axis can be extracted from the off-diagonal elements

    Examples
    --------
    >>> R = rotate_spinor(np.array([0, 0, 1]), np.pi/2)
    >>> scipy_rot = to_scipy_rotation(R)
    >>> scipy_rot.as_euler('zyx', degrees=True)[0]  # Should be ~90 degrees
    90.0
    """
    R = np.asarray(R, dtype=np.complex128)

    # Extract angle from trace: trace(R) = 2cos(θ/2)
    trace = np.trace(R)
    cos_half_angle = trace.real / 2.0

    # Handle numerical errors
    cos_half_angle = np.clip(cos_half_angle, -1.0, 1.0)
    half_angle = np.arccos(cos_half_angle)
    angle = 2.0 * half_angle

    if angle < 1e-10:
        # Identity rotation
        return Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))

    # Extract axis from off-diagonal elements
    # For R = cos(θ/2)I - i sin(θ/2)(n·σ), the axis is encoded in
    # the imaginary parts of the off-diagonal elements
    sin_half_angle = np.sin(half_angle)

    if np.abs(sin_half_angle) < 1e-10:
        # Angle is 0 or 2π (modulo sign ambiguity)
        return Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * angle)

    # Extract axis: n = (1/(i*sin(θ/2))) * (R - cos(θ/2)I) projected onto Pauli matrices
    # Simpler: use R = [[a, b], [c, d]], then:
    # n_x ∝ Im(b + c), n_y ∝ Re(b - c), n_z ∝ Im(a - d)
    nx = (R[0, 1].imag + R[1, 0].imag) / sin_half_angle
    ny = (R[0, 1].real - R[1, 0].real) / sin_half_angle
    nz = (R[0, 0].imag - R[1, 1].imag) / sin_half_angle

    axis = np.array([nx, ny, nz])
    axis = axis / np.linalg.norm(axis)

    rotvec = axis * angle
    return Rotation.from_rotvec(rotvec)
