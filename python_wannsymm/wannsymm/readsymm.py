"""
Symmetry reading module for WannSymm

Reads symmetry operations from files and interfaces with spglib.
Translated from: src/readsymm.h and src/readsymm.c

Translation Status: ✅ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from .readinput import parseline, str2boolean, InputParseError
from .matrix import matrix3x3_determinant


class SymmetryError(Exception):
    """Exception raised when symmetry operations are invalid."""
    pass


@dataclass
class SymmetryOperation:
    """
    Single symmetry operation.
    
    Attributes
    ----------
    rotation : np.ndarray
        3x3 rotation matrix (must be orthogonal with det = ±1)
    translation : np.ndarray
        3-element translation vector (fractional coordinates)
    time_reversal : bool
        Whether this operation includes time-reversal
    """
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    time_reversal: bool
    
    def __post_init__(self):
        """Validate symmetry operation after initialization."""
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64)
        
        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {self.rotation.shape}")
        if self.translation.shape != (3,):
            raise ValueError(f"Translation vector must have 3 elements, got {self.translation.shape}")


@dataclass
class SymmetryData:
    """
    Container for all symmetry operations.
    
    Attributes
    ----------
    operations : List[SymmetryOperation]
        List of symmetry operations
    global_time_reversal : bool
        Whether global time-reversal symmetry is present
    """
    operations: List[SymmetryOperation]
    global_time_reversal: bool


def validate_rotation_matrix(
    rotation: npt.NDArray[np.float64],
    tol: float = 1e-6
) -> None:
    """
    Validate that a rotation matrix is orthogonal with determinant ±1.
    
    Checks:
    1. Matrix is orthogonal: R^T @ R = I
    2. Determinant is ±1
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix to validate
    tol : float, optional
        Tolerance for validation checks (default: 1e-6)
        
    Raises
    ------
    SymmetryError
        If rotation matrix is not valid
        
    Examples
    --------
    >>> rot = np.eye(3)  # Identity is a valid rotation
    >>> validate_rotation_matrix(rot)
    
    >>> rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    >>> validate_rotation_matrix(rot)  # 90-degree rotation around z
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    
    if rotation.shape != (3, 3):
        raise SymmetryError(f"Rotation matrix must be 3x3, got {rotation.shape}")
    
    # Check orthogonality: R^T @ R should equal identity
    rt_r = rotation.T @ rotation
    identity = np.eye(3)
    
    if not np.allclose(rt_r, identity, atol=tol):
        max_diff = np.max(np.abs(rt_r - identity))
        raise SymmetryError(
            f"Rotation matrix is not orthogonal. "
            f"R^T @ R deviates from identity by {max_diff:.2e}"
        )
    
    # Check determinant is ±1
    det = matrix3x3_determinant(rotation)
    if not np.isclose(abs(det), 1.0, atol=tol):
        raise SymmetryError(
            f"Rotation matrix determinant is {det:.6f}, "
            f"should be ±1"
        )


def readsymm(
    filename: str,
    validate: bool = True,
    tol: float = 1e-6
) -> SymmetryData:
    """
    Read symmetry operations from file.
    
    Equivalent to C function: void readsymm(char * fn_symm, double rotations[][3][3],
                                            double translations[][3], int TR[], 
                                            int * p2nsymm, int * p2flag_trsymm)
    
    File format:
    - Lines with 'global_trsymm', 'trsymm', or 'globaltime-reversalsymmetry' tags
      set the global time-reversal symmetry flag
    - Line with 'nsymm' tag specifies number of symmetries
    - For each symmetry:
      * Optional line with 'angle' tag (not fully implemented in C version)
      * Three lines with 3x3 rotation matrix (space-separated floats)
      * One line with translation vector (3 floats) followed by optional T/F flag
        for time-reversal symmetry of this operation
    
    Parameters
    ----------
    filename : str
        Path to symmetry file
    validate : bool, optional
        Whether to validate rotation matrices (default: True)
    tol : float, optional
        Tolerance for validation (default: 1e-6)
        
    Returns
    -------
    SymmetryData
        Parsed symmetry operations with global time-reversal flag
        
    Raises
    ------
    InputParseError
        If file cannot be parsed
    SymmetryError
        If symmetry operations are invalid
        
    Examples
    --------
    >>> # Read symmetries from file
    >>> symm_data = readsymm("symmetries.dat")
    >>> print(f"Number of symmetries: {len(symm_data.operations)}")
    >>> print(f"Global TR symmetry: {symm_data.global_time_reversal}")
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        raise InputParseError(f"Cannot open symmetry file '{filename}': {e}")
    
    # Parse header to find nsymm and global_trsymm
    global_trsymm = True  # Default value
    nsymm = None
    line_idx = 0
    
    while line_idx < len(lines):
        line = lines[line_idx]
        tag, arg = parseline(line, ignorecase=True)
        
        if tag in ['globaltime-reversalsymmetry', 'trsymm', 'global_trsymm']:
            value = str2boolean(arg)
            if value is None:
                raise InputParseError(
                    f"Invalid boolean value for {tag}: '{arg}'. "
                    f"Should be T/F, True/False, or 1/0",
                    line_number=line_idx + 1,
                    line=line
                )
            global_trsymm = value
        elif tag == 'nsymm':
            try:
                nsymm = int(arg)
            except ValueError:
                raise InputParseError(
                    f"Invalid integer value for nsymm: '{arg}'",
                    line_number=line_idx + 1,
                    line=line
                )
            line_idx += 1
            break
        
        line_idx += 1
    
    if nsymm is None:
        raise InputParseError("Missing 'nsymm' tag in symmetry file")
    
    if nsymm < 1:
        raise InputParseError(f"nsymm must be positive, got {nsymm}")
    
    # Parse symmetry operations
    operations = []
    
    for i in range(nsymm):
        if line_idx >= len(lines):
            raise InputParseError(
                f"Unexpected end of file while reading symmetry {i+1}/{nsymm}"
            )
        
        # Skip separator lines like "--- N ---"
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line and not line.startswith('---'):
                break
            line_idx += 1
        
        if line_idx >= len(lines):
            raise InputParseError(
                f"Unexpected end of file while reading symmetry {i+1}/{nsymm}"
            )
        
        # Check for optional 'angle' tag (not fully supported yet)
        line = lines[line_idx]
        tag, arg = parseline(line, ignorecase=True)
        
        if tag == 'angle':
            # Skip angle-based input (not implemented in C version either)
            raise InputParseError(
                "Angle-based symmetry input not yet supported",
                line_number=line_idx + 1,
                line=line
            )
        
        # Read 3x3 rotation matrix
        rotation = np.zeros((3, 3), dtype=np.float64)
        try:
            for j in range(3):
                if line_idx >= len(lines):
                    raise InputParseError(
                        f"Unexpected end of file while reading rotation matrix "
                        f"for symmetry {i+1}"
                    )
                values = lines[line_idx].split()
                if len(values) < 3:
                    raise InputParseError(
                        f"Expected 3 values for rotation matrix row {j+1}, got {len(values)}",
                        line_number=line_idx + 1,
                        line=lines[line_idx]
                    )
                rotation[j, :] = [float(values[0]), float(values[1]), float(values[2])]
                line_idx += 1
        except (ValueError, IndexError) as e:
            raise InputParseError(
                f"Error parsing rotation matrix for symmetry {i+1}: {e}",
                line_number=line_idx + 1
            )
        
        # Validate rotation matrix if requested
        if validate:
            try:
                validate_rotation_matrix(rotation, tol=tol)
            except SymmetryError as e:
                raise SymmetryError(
                    f"Invalid rotation matrix for symmetry {i+1}: {e}"
                )
        
        # Read translation vector and time-reversal flag
        if line_idx >= len(lines):
            raise InputParseError(
                f"Unexpected end of file while reading translation "
                f"for symmetry {i+1}"
            )
        
        line = lines[line_idx]
        parts = line.split()
        
        if len(parts) < 3:
            raise InputParseError(
                f"Expected at least 3 values for translation, got {len(parts)}",
                line_number=line_idx + 1,
                line=line
            )
        
        try:
            translation = np.array([float(parts[0]), float(parts[1]), float(parts[2])],
                                   dtype=np.float64)
        except ValueError as e:
            raise InputParseError(
                f"Error parsing translation for symmetry {i+1}: {e}",
                line_number=line_idx + 1,
                line=line
            )
        
        # Parse time-reversal flag from the same line
        # Check if line contains 'T' or 't' but not 'F' or 'f'
        # Logic from C code:
        # if ( strchr(line, 'T') ==NULL && strchr(line, 't') == NULL )
        #     TR[i] = 0;
        # else if(strchr(line, 'F') !=NULL && strchr(line, 'f') != NULL)  [sic - AND not OR]
        #     TR[i] = 0;
        # else
        #     TR[i] = 1;
        
        time_reversal = False
        line_upper = line.upper()
        has_t = 'T' in line_upper
        has_f = 'F' in line_upper
        
        # Based on C logic: TR is True if 'T' is present and not both 'F' and 'f'
        # The C code has a bug (uses AND instead of OR for the F check)
        # We'll match the C behavior for compatibility
        if has_t and not (has_f and 'f' in line):
            time_reversal = True
            # If any symmetry has time-reversal, global time-reversal is disabled
            global_trsymm = False
        
        operations.append(SymmetryOperation(
            rotation=rotation,
            translation=translation,
            time_reversal=time_reversal
        ))
        
        line_idx += 1
    
    return SymmetryData(
        operations=operations,
        global_time_reversal=global_trsymm
    )
