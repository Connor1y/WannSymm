"""
Basis rotation module for WannSymm

Combines orbital and spinor rotations for complete basis transformation.
Translated from: src/rotate_basis.h and src/rotate_basis.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from .rotate_orbital import rotate_cubic
from .rotate_spinor import rotate_spinor
from .matrix import matrix3x3_inverse, matrix3x3_dot, matrix3x3_transpose

# Type aliases
ComplexMatrix = npt.NDArray[np.complex128]
RealVector = npt.NDArray[np.float64]


def get_axis_angle_of_rotation(
    rotation: npt.NDArray[np.float64],
    lattice: npt.NDArray[np.float64],
    eps: float = 1e-7
) -> Tuple[npt.NDArray[np.float64], float, int]:
    """
    Extract rotation axis, angle, and inversion flag from a rotation matrix.
    
    Converts a rotation matrix in fractional coordinates to axis-angle form
    in Cartesian coordinates, with special handling for improper rotations.
    
    Equivalent to C function: void get_axis_angle_of_rotation(double axis[3], 
                                   double * angle, int * inv, double rin[3][3], 
                                   double lattice[3][3])
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix in fractional coordinates
    lattice : np.ndarray
        3x3 lattice matrix (row vectors)
    eps : float, optional
        Tolerance for numerical comparisons (default: 1e-7)
        
    Returns
    -------
    axis : np.ndarray
        Rotation axis as a unit vector (3,)
    angle : float
        Rotation angle in radians
    inv : int
        Inversion flag (1 if improper rotation, 0 if proper)
        
    Notes
    -----
    The function:
    1. Converts rotation from fractional to Cartesian coordinates
    2. Extracts determinant to check for improper rotations (det = -1)
    3. Computes axis and angle using the trace and off-diagonal elements
    4. Normalizes the axis to unit length
    
    For improper rotations (mirrors, rotoreflections), the determinant is -1.
    The function returns the rotation part after removing the inversion.
    """
    # Convert lattice to column-major and get inverse
    lattice_T = lattice.T  # Column-major
    inv_lattice = matrix3x3_inverse(lattice_T)
    
    # Transform rotation to Cartesian: R_cart = A * R * A^-1
    tmp = matrix3x3_dot(lattice_T, rotation)
    rotation_cartesian = matrix3x3_dot(tmp, inv_lattice)
    
    # Compute determinant
    determinant = np.linalg.det(rotation_cartesian)
    
    # Check for inversion
    inv_flag = 1 if determinant < 0 else 0
    
    # Make it a proper rotation by multiplying by sign(det)
    sign_det = 1.0 if determinant >= 0 else -1.0
    rot = rotation_cartesian * sign_det
    
    # Determine type of rotation
    trace = np.trace(rot)
    
    # Check for identity
    is_identity = (
        abs(rot[0, 0] - 1.0) < eps and
        abs(rot[1, 1] - 1.0) < eps and
        abs(rot[2, 2] - 1.0) < eps and
        abs(rot[0, 1]) < eps and abs(rot[1, 2]) < eps and abs(rot[0, 2]) < eps and
        abs(rot[1, 0]) < eps and abs(rot[2, 1]) < eps and abs(rot[2, 0]) < eps
    )
    
    if is_identity:
        # Identity rotation
        return np.array([0.0, 0.0, 1.0]), 0.0, inv_flag
    
    # Check for 180 degree rotation (eigenvalue -1)
    det1 = (
        (rot[0, 0] + 1.0) * ((rot[1, 1] + 1.0) * (rot[2, 2] + 1.0) - rot[2, 1] * rot[1, 2]) -
        rot[0, 1] * (rot[1, 0] * (rot[2, 2] + 1.0) - rot[2, 0] * rot[1, 2]) +
        rot[0, 2] * (rot[1, 0] * rot[2, 1] - rot[2, 0] * (rot[1, 1] + 1.0))
    )
    
    if abs(det1) < eps:
        # 180 degree rotation
        angle = np.pi
        
        # First try to find axis parallel to coordinate axis
        axis = np.zeros(3)
        for i in range(3):
            if abs(rot[i, i] - 1.0) < eps:
                axis[i] = 1.0
        
        norm = np.linalg.norm(axis)
        if norm < eps:
            # General case for 180 rotation
            for i in range(3):
                axis[i] = np.sqrt(abs(rot[i, i] + 1.0) / 2.0)
            
            # Use off-diagonal elements to determine signs
            for i in range(3):
                for j in range(i + 1, 3):
                    if abs(axis[i] * axis[j]) > eps:
                        axis[i] = 0.5 * rot[i, j] / axis[j]
    else:
        # General rotation (not 0 or 180 degrees)
        # Extract axis from antisymmetric part
        axis = np.array([
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1]
        ])
        
        norm = np.linalg.norm(axis)
        if norm > eps:
            axis = axis / norm
        else:
            axis = np.array([0.0, 0.0, 1.0])
        
        # Compute angle from trace: trace = 1 + 2*cos(angle)
        angle = np.arccos((trace - 1.0) / 2.0)
        
        # Determine sign of angle using off-diagonal elements
        # Check consistency with rotation formula
        test_value = axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle)
        if abs(test_value - rot[1, 0]) > 1e-3:
            angle = -angle
    
    # Normalize axis direction (prefer positive z, x, or y)
    if axis[2] < -eps:
        axis = -axis
        angle = -angle
    elif abs(axis[2]) < eps and axis[0] < -eps:
        axis = -axis
        angle = -angle
    elif abs(axis[2]) < eps and abs(axis[0]) < eps and axis[1] < -eps:
        axis = -axis
        angle = -angle
    
    # Final normalization
    norm = np.linalg.norm(axis)
    if norm > eps:
        axis = axis / norm
    else:
        axis = np.array([0.0, 0.0, 1.0])
    
    # Adjust angle to be in range (-π, π]
    if angle < -np.pi + 1e-3:
        angle += 2 * np.pi
    
    return axis, angle, inv_flag


def combine_basis_rotation(
    orbital_rotations: List[ComplexMatrix],
    spin_rotation: ComplexMatrix,
    orbital_indices: npt.NDArray[np.int32],
    flag_soc: bool = False
) -> ComplexMatrix:
    """
    Combine orbital and spin rotations into full basis rotation matrix.
    
    This function constructs the full unitary transformation matrix for
    rotating the Wannier orbital basis. The structure depends on whether
    spin-orbit coupling (SOC) is included.
    
    IMPORTANT: orbital_indices represents the l quantum number for each
    WANNIER ORBITAL (not each basis function component). For example:
    - [1, 1, 1] means 3 p-orbitals (each with 2l+1=3 components: px, py, pz)
    - Total basis size would be 9 (3 orbitals × 3 components each)
    
    The orbitals are grouped by l-shell, and each complete l-shell rotates
    together using the appropriate rotation matrix for that l.
    
    - Non-SOC case: Block-diagonal orbital rotations
      Basis ordering: all components of orbital 1, all of orbital 2, etc.
      
    - SOC case: Tensor product orbital ⊗ spin  
      Basis ordering (C convention): spin varies fastest within each orbital component
      |orb1_comp1, ↑⟩, |orb1_comp1, ↓⟩, |orb1_comp2, ↑⟩, |orb1_comp2, ↓⟩, ...
    
    Parameters
    ----------
    orbital_rotations : list of np.ndarray
        List of orbital rotation matrices, one for each l value (l=0,1,2,3).
        Each matrix has shape (2l+1, 2l+1).
    spin_rotation : np.ndarray
        Spin rotation matrix, shape (2, 2).
    orbital_indices : np.ndarray
        Array of l quantum numbers, one per WANNIER ORBITAL.
        Each orbital contributes (2l+1) basis functions.
    flag_soc : bool, optional
        If True, include spin-orbit coupling.
    
    Returns
    -------
    np.ndarray
        Full basis rotation matrix.
    """
    norb = len(orbital_indices)
    
    # Calculate total number of basis functions
    # Each orbital with l contributes (2l+1) components
    total_basis_funcs = sum(2 * l + 1 for l in orbital_indices)
    
    if not flag_soc:
        # Non-SOC case: Block-diagonal orbital rotations
        basis_rotation = np.zeros((total_basis_funcs, total_basis_funcs), dtype=np.complex128)
        
        offset = 0
        for l in orbital_indices:
            dim = 2 * l + 1
            orb_rot = orbital_rotations[l]
            
            # Place this orbital's rotation matrix in the diagonal block
            basis_rotation[offset:offset+dim, offset:offset+dim] = orb_rot
            offset += dim
        
        return basis_rotation
    
    else:
        # SOC case: Tensor product orbital ⊗ spin
        # Total dimension includes spin
        full_dim = 2 * total_basis_funcs
        basis_rotation = np.zeros((full_dim, full_dim), dtype=np.complex128)
        
        # Process each orbital
        offset = 0
        for l in orbital_indices:
            dim = 2 * l + 1
            orb_rot = orbital_rotations[l]
            
            # For this orbital, create tensor product with spin
            # The block for this orbital has size (2*dim) × (2*dim)
            block_size = 2 * dim
            block_start = 2 * offset
            
            # Fill in the block as Kronecker product: orb ⊗ spin
            # With C ordering (spin varies fastest)
            for i in range(dim):
                for j in range(dim):
                    orb_elem = orb_rot[i, j]
                    
                    # This orbital element couples to spin states
                    for si in range(2):
                        for sj in range(2):
                            spin_elem = spin_rotation[si, sj]
                            combined = orb_elem * spin_elem
                            
                            # C ordering: spin varies fastest
                            row = block_start + 2 * i + si
                            col = block_start + 2 * j + sj
                            
                            basis_rotation[row, col] = combined
            
            offset += dim
        
        return basis_rotation


def get_basis_rotation(
    l_values: List[int],
    axis: RealVector,
    angle: float,
    inv: bool = False,
    flag_soc: bool = False
) -> ComplexMatrix:
    """
    Get full basis rotation matrix for given rotation parameters.
    
    Convenience function that generates orbital and spin rotations
    and combines them into the full basis rotation.
    
    Parameters
    ----------
    l_values : list of int
        List of angular momentum quantum numbers, one per Wannier orbital.
        Each orbital contributes (2l+1) basis functions.
        Example: [1, 1, 1] means 3 p-orbitals, total of 9 basis functions.
    axis : array_like
        Rotation axis as 3D vector, shape (3,).
    angle : float
        Rotation angle in radians.
    inv : bool, optional
        Inversion flag for orbital rotation. Default is False.
    flag_soc : bool, optional
        If True, include spin-orbit coupling. Default is False.
    
    Returns
    -------
    np.ndarray
        Full basis rotation matrix.
    
    Examples
    --------
    >>> # Non-SOC rotation for 3 p-orbitals (9 basis functions)
    >>> rot = get_basis_rotation([1, 1, 1], [0, 0, 1], np.pi/2, flag_soc=False)
    >>> rot.shape
    (9, 9)
    
    >>> # SOC rotation for same orbitals (18 = 9 × 2 for spin)
    >>> rot_soc = get_basis_rotation([1, 1, 1], [0, 0, 1], np.pi/2, flag_soc=True)
    >>> rot_soc.shape
    (18, 18)
    """
    # Generate orbital rotations for all l values
    orbital_rotations = []
    for l in range(4):  # l = 0, 1, 2, 3
        orb_rot = rotate_cubic(l, axis, angle, inv)
        orbital_rotations.append(orb_rot)
    
    # Generate spin rotation
    if flag_soc:
        spin_rot = rotate_spinor(axis, angle, inv=0)
    else:
        # For non-SOC, use identity (though it won't be used)
        spin_rot = np.eye(2, dtype=np.complex128)
    
    # Convert l_values to array of indices
    orbital_indices = np.array(l_values, dtype=np.int32)
    
    # Combine into full basis rotation
    basis_rot = combine_basis_rotation(
        orbital_rotations, spin_rot, orbital_indices, flag_soc
    )
    
    return basis_rot
