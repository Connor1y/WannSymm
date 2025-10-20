"""
Orbital rotation module for WannSymm

Rotates orbital basis functions using Wigner D-matrices.
Translated from: src/rotate_orbital.h and src/rotate_orbital.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from scipy import linalg
from functools import lru_cache

# Type aliases
ComplexMatrix = npt.NDArray[np.complex128]
RealVector = npt.NDArray[np.float64]


def generate_Lmatrix(l: int) -> ComplexMatrix:
    """
    Generate the angular momentum operator matrices (Lx, Ly, Lz) in |lm> basis.
    
    This function generates the L operator in the |lm> basis, which combined 
    with real-space harmonics can be used for spatial rotation generation.
    Assumes ℏ = 1.
    
    Equivalent to C function: void generate_Lmatrix(dcomplex * Lmat, int l)
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number
        l=0: s orbitals
        l=1: p orbitals  
        l=2: d orbitals
        l=3: f orbitals
    
    Returns
    -------
    np.ndarray
        3D array of shape (3, 2l+1, 2l+1) containing [Lx, Ly, Lz] matrices
        
    Notes
    -----
    The angular momentum ladder operators are defined as:
    - L+ = Lx + i*Ly
    - L- = Lx - i*Ly
    - L+|lm> = sqrt[(l-m)(l+m+1)] |l,m+1>
    - L-|lm> = sqrt[(l+m)(l-m+1)] |l,m-1>
    - Lz|lm> = m |lm>
    
    Then:
    - Lx = (L+ + L-)/2
    - Ly = (L+ - L-)/(2i)
    - Lz is diagonal with eigenvalues m
    """
    N = 2 * l + 1
    
    # Initialize ladder operators
    lp = np.zeros((N, N), dtype=np.complex128)  # L+ = Lx + i*Ly
    lm = np.zeros((N, N), dtype=np.complex128)  # L- = Lx - i*Ly
    lz = np.zeros((N, N), dtype=np.complex128)  # Lz
    
    for m in range(-l, l + 1):
        idx = m + l  # Convert m to array index (0 to 2l)
        
        # Lz|lm> = m |lm>
        lz[idx, idx] = m
        
        # L+|lm> = sqrt[(l-m)(l+m+1)] |l,m+1>
        if m < l:
            lp[idx + 1, idx] = np.sqrt((l - m) * (l + m + 1))
        
        # L-|lm> = sqrt[(l+m)(l-m+1)] |l,m-1>
        if m > -l:
            lm[idx - 1, idx] = np.sqrt((l + m) * (l - m + 1))
    
    # Construct Lx, Ly, Lz from ladder operators
    Lmat = np.zeros((3, N, N), dtype=np.complex128)
    Lmat[0] = (lp + lm) / 2.0              # Lx = (L+ + L-)/2
    Lmat[1] = (lp - lm) / (2.0j)           # Ly = (L+ - L-)/(2i)
    Lmat[2] = lz                            # Lz
    
    return Lmat


def rotate_Ylm(l: int, axis: RealVector, alpha: float, inv: bool = False) -> ComplexMatrix:
    """
    Generate rotation matrix for spherical harmonics Y_lm with specific l.
    
    This function computes the Wigner D-matrix for rotating spherical harmonics
    by angle alpha around the given axis using the exponential formula:
    R = exp(-i*alpha*n·L)
    
    Equivalent to C function: void rotate_Ylm(dcomplex * rot, int l, 
                                              double axis[3], double alpha, int inv)
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number (0, 1, 2, or 3)
    axis : array_like
        Rotation axis as 3D unit vector [x, y, z]
    alpha : float
        Rotation angle in radians
    inv : bool, optional
        If True and l is odd, multiply result by -1 (for inversion symmetry)
        Default is False
    
    Returns
    -------
    np.ndarray
        Rotation matrix of shape (2l+1, 2l+1) for spherical harmonics
        
    Notes
    -----
    The rotation is computed using matrix exponential:
    1. Compute -i*alpha*(n·L) where n is the normalized axis
    2. Diagonalize this matrix: M = V*D*V†
    3. Compute exp(M) = V*exp(D)*V†
    """
    N = 2 * l + 1
    
    # Normalize axis
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_norm = axis_array / np.linalg.norm(axis_array)
    
    # Generate angular momentum matrices
    Lmat = generate_Lmatrix(l)
    
    # Calculate -i*alpha*n·L
    Ldotn = -1j * alpha * (axis_norm[0] * Lmat[0] + 
                            axis_norm[1] * Lmat[1] + 
                            axis_norm[2] * Lmat[2])
    
    # Diagonalize to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(Ldotn)
    
    # Compute exp(eigenvalues)
    exp_eig = np.exp(eigenvalues)
    
    # Reconstruct rotation matrix: R = V * exp(D) * V†
    rot = eigenvectors @ np.diag(exp_eig) @ eigenvectors.conj().T
    
    # Apply inversion factor if needed
    if inv and (l % 2 == 1):
        rot = -rot
    
    return rot


def generate_C2Ylm(l: int) -> Tuple[Optional[ComplexMatrix], Optional[ComplexMatrix]]:
    """
    Generate transformation matrices between cubic and spherical harmonics.
    
    Cubic harmonics are real linear combinations of spherical harmonics that
    transform according to cubic symmetry. This function generates the 
    transformation matrices to convert between the two representations.
    
    Equivalent to C function: void generate_C2Ylm(dcomplex * C2Ylm, 
                                                   dcomplex * Ylm2C, int l)
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number (0, 1, 2, or 3)
    
    Returns
    -------
    C2Ylm : np.ndarray
        Transformation matrix from cubic to spherical harmonics (2l+1, 2l+1)
    Ylm2C : np.ndarray
        Transformation matrix from spherical to cubic harmonics (2l+1, 2l+1)
        
    Notes
    -----
    Definitions and ordering of cubic harmonics:
    
    l=0 (s):
      s = |0,0>
      
    l=1 (p):
      pz = |1,0>
      px = (|1,-1> - |1,1>)/√2
      py = (|1,-1> + |1,1>)*i/√2
      
    l=2 (d):
      dz² = |2,0>
      dzx = (|2,-1> - |2,1>)/√2
      dzy = (|2,-1> + |2,1>)*i/√2
      dx²-y² = (|2,-2> + |2,2>)/√2
      dxy = (|2,-2> - |2,2>)*i/√2
      
    l=3 (f):
      fz³ = |3,0>
      fxz² = (|3,-1> - |3,1>)/√2
      fyz² = (|3,-1> + |3,1>)*i/√2
      fz(x²-y²) = (|3,-2> + |3,2>)/√2
      fxyz = (|3,-2> - |3,2>)*i/√2
      fx(x²-3y²) = (|3,-3> - |3,3>)/√2
      fy(3x²-y²) = (|3,-3> + |3,3>)*i/√2
    """
    N = 2 * l + 1
    sqrt2 = np.sqrt(2.0)
    
    # Initialize Ylm2C (spherical to cubic)
    Ylm2C = np.zeros((N, N), dtype=np.complex128)
    
    if l == 0:
        # s orbital
        Ylm2C[0, 0] = 1.0
        
    elif l == 1:
        # p orbitals: pz, px, py
        Ylm2C[0, 1] = 1.0                      # pz = |1,0>
        Ylm2C[1, 0] = 1.0 / sqrt2              # px = (|1,-1> - |1,1>)/√2
        Ylm2C[1, 2] = -1.0 / sqrt2
        Ylm2C[2, 0] = 1.0j / sqrt2             # py = (|1,-1> + |1,1>)*i/√2
        Ylm2C[2, 2] = 1.0j / sqrt2
        
    elif l == 2:
        # d orbitals: dz², dzx, dzy, dx²-y², dxy
        Ylm2C[0, 2] = 1.0                      # dz² = |2,0>
        Ylm2C[1, 1] = 1.0 / sqrt2              # dzx = (|2,-1> - |2,1>)/√2
        Ylm2C[1, 3] = -1.0 / sqrt2
        Ylm2C[2, 1] = 1.0j / sqrt2             # dzy = (|2,-1> + |2,1>)*i/√2
        Ylm2C[2, 3] = 1.0j / sqrt2
        Ylm2C[3, 0] = 1.0 / sqrt2              # dx²-y² = (|2,-2> + |2,2>)/√2
        Ylm2C[3, 4] = 1.0 / sqrt2
        Ylm2C[4, 0] = 1.0j / sqrt2             # dxy = (|2,-2> - |2,2>)*i/√2
        Ylm2C[4, 4] = -1.0j / sqrt2
        
    elif l == 3:
        # f orbitals
        Ylm2C[0, 3] = 1.0                      # fz³ = |3,0>
        Ylm2C[1, 2] = 1.0 / sqrt2              # fxz² = (|3,-1> - |3,1>)/√2
        Ylm2C[1, 4] = -1.0 / sqrt2
        Ylm2C[2, 2] = 1.0j / sqrt2             # fyz² = (|3,-1> + |3,1>)*i/√2
        Ylm2C[2, 4] = 1.0j / sqrt2
        Ylm2C[3, 1] = 1.0 / sqrt2              # fz(x²-y²) = (|3,-2> + |3,2>)/√2
        Ylm2C[3, 5] = 1.0 / sqrt2
        Ylm2C[4, 1] = 1.0j / sqrt2             # fxyz = (|3,-2> - |3,2>)*i/√2
        Ylm2C[4, 5] = -1.0j / sqrt2
        Ylm2C[5, 0] = 1.0 / sqrt2              # fx(x²-3y²) = (|3,-3> - |3,3>)/√2
        Ylm2C[5, 6] = -1.0 / sqrt2
        Ylm2C[6, 0] = 1.0j / sqrt2             # fy(3x²-y²) = (|3,-3> + |3,3>)*i/√2
        Ylm2C[6, 6] = 1.0j / sqrt2
    else:
        raise ValueError(f"l={l} not supported. Only l=0,1,2,3 are implemented.")
    
    # Compute C2Ylm as inverse of Ylm2C
    C2Ylm = linalg.inv(Ylm2C)
    
    return C2Ylm, Ylm2C


def rotate_cubic(l: int, axis: RealVector, alpha: float, inv: bool = False) -> ComplexMatrix:
    """
    Generate rotation matrix for cubic harmonics with specific l.
    
    Cubic harmonics are rotated by transforming to spherical harmonics,
    applying the spherical rotation, then transforming back:
    D_cubic = Ylm2C · D_sphere · C2Ylm
    
    Equivalent to C function: void rotate_cubic(dcomplex * rot, int l, 
                                                double axis[3], double alpha, int inv)
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number (0, 1, 2, or 3)
    axis : array_like
        Rotation axis as 3D unit vector [x, y, z]
    alpha : float
        Rotation angle in radians
    inv : bool, optional
        If True and l is odd, multiply result by -1 (for inversion symmetry)
        Default is False
    
    Returns
    -------
    np.ndarray
        Rotation matrix of shape (2l+1, 2l+1) for cubic harmonics
        
    Notes
    -----
    The transformation follows: Cubic -> Spherical -> Rotate -> Cubic
    This ensures that the rotation respects the cubic harmonic basis.
    The formula is: |cubic'> = Ylm2C @ D_sphere @ C2Ylm @ |cubic>
    """
    # Get transformation matrices
    C2Ylm, Ylm2C = generate_C2Ylm(l)
    
    # Get spherical harmonics rotation
    rot_Ylm = rotate_Ylm(l, axis, alpha, inv)
    
    # Transform: D_cubic = Ylm2C · D_sphere · C2Ylm
    # This transforms cubic basis -> spherical, rotates, then transforms back to cubic
    rot = Ylm2C @ rot_Ylm @ C2Ylm
    
    return rot


@lru_cache(maxsize=128)
def _cached_rotate_cubic(l: int, axis_tuple: Tuple[float, float, float], 
                         alpha: float, inv: bool = False) -> ComplexMatrix:
    """
    Cached version of rotate_cubic for performance.
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number
    axis_tuple : tuple of float
        Rotation axis as tuple (x, y, z) - must be hashable for caching
    alpha : float
        Rotation angle in radians
    inv : bool
        Inversion flag
    
    Returns
    -------
    np.ndarray
        Rotation matrix for cubic harmonics
    """
    axis = np.array(axis_tuple, dtype=np.float64)
    return rotate_cubic(l, axis, alpha, inv)


def get_rotation_matrix(l: int, axis: RealVector, alpha: float, 
                        inv: bool = False, use_cache: bool = True) -> ComplexMatrix:
    """
    Get rotation matrix for cubic harmonics with optional caching.
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number (0, 1, 2, or 3)
    axis : array_like
        Rotation axis as 3D vector (will be normalized)
    alpha : float
        Rotation angle in radians
    inv : bool, optional
        If True and l is odd, multiply result by -1
    use_cache : bool, optional
        If True, use cached results for performance. Default is True.
    
    Returns
    -------
    np.ndarray
        Rotation matrix of shape (2l+1, 2l+1)
    """
    if use_cache:
        axis_array = np.asarray(axis, dtype=np.float64)
        axis_norm = axis_array / np.linalg.norm(axis_array)
        axis_tuple = tuple(axis_norm)
        return _cached_rotate_cubic(l, axis_tuple, alpha, inv)
    else:
        return rotate_cubic(l, axis, alpha, inv)
