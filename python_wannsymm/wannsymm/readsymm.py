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


def derive_symm_for_magnetic_materials(
    rotations: List[npt.NDArray[np.float64]],
    translations: List[npt.NDArray[np.float64]],
    lattice: npt.NDArray[np.float64],
    atom_positions: npt.NDArray[np.float64],
    atom_types: List[int],
    magmom: npt.NDArray[np.float64],
    symm_magnetic_tolerance: float = 1e-3
) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]], List[bool]]:
    """
    Derive symmetries for magnetic materials.
    
    This function filters crystal symmetries based on how they transform magnetic moments.
    Only symmetries that preserve the magnetic order (possibly with time-reversal) are kept.
    
    Equivalent to C function: void derive_symm_for_magnetic_materials(...)
    in src/readinput.c
    
    Parameters
    ----------
    rotations : List[np.ndarray]
        List of 3x3 rotation matrices (fractional coordinates)
    translations : List[np.ndarray]
        List of 3-element translation vectors (fractional coordinates)
    lattice : np.ndarray
        3x3 lattice matrix (row vectors in Cartesian coordinates)
    atom_positions : np.ndarray
        Nx3 array of atomic positions (fractional coordinates)
    atom_types : List[int]
        List of atom type indices for each atom
    magmom : np.ndarray
        Nx3 array of magnetic moments (Cartesian coordinates)
    symm_magnetic_tolerance : float
        Tolerance for comparing magnetic moments
        
    Returns
    -------
    filtered_rotations : List[np.ndarray]
        Filtered rotation matrices
    filtered_translations : List[np.ndarray]
        Filtered translation vectors
    TR_flags : List[bool]
        Time-reversal flags for each filtered symmetry
        
    Notes
    -----
    For each symmetry operation S=(R,t):
    1. Transform each atom: r' = R·r + t
    2. Transform each magnetic moment: m' = R_cart·m (with sign flip for inversion)
    3. If m'_transformed ≈ m_final for all atoms: keep symmetry with TR=False
    4. If m'_transformed ≈ -m_final for all atoms: keep symmetry with TR=True
    5. Otherwise: discard symmetry
    """
    from .vector import Vector, vector_norm, vector_sub, vector_round
    
    natom = len(atom_positions)
    nsymm_crystal = len(rotations)
    
    # Convert lattice for transformations
    # latt_trans = transpose(lattice) for fractional->Cartesian
    latt_trans = lattice.T  
    latt_trans_inv = np.linalg.inv(latt_trans)
    
    # Store filtered symmetries
    filtered_rotations = []
    filtered_translations = []
    TR_flags = []
    
    # Check each symmetry operation
    for isymm in range(nsymm_crystal):
        rot = rotations[isymm]
        trans = translations[isymm]
        
        # Compute Cartesian rotation: R_cart = latt_trans · R · latt_trans^-1
        rot_cart = latt_trans @ rot @ latt_trans_inv
        
        keep_symm_vote = 0
        tr_vote = 0
        tot_votes = 0
        
        # Check each magnetic atom
        for iatom in range(natom):
            # Skip atoms with zero magnetic moment (no vote)
            mag_norm = np.linalg.norm(magmom[iatom])
            if mag_norm < abs(symm_magnetic_tolerance):
                continue
            
            tot_votes += 1
            
            # Transform atomic position: r' = R·r + t
            atpos_symmed = rot @ atom_positions[iatom] + trans
            
            # Transform magnetic moment: m' = R_cart·m
            mag_roted = rot_cart @ magmom[iatom]
            
            # If inversion (det(R) < 0), flip sign of rotated moment
            # Inversion doesn't affect spin space
            if matrix3x3_determinant(rot) < 0:
                mag_roted = -mag_roted
            
            # Find which atom the transformed position corresponds to
            jatom = None
            for j in range(natom):
                # Distance in fractional coordinates
                dis = atpos_symmed - atom_positions[j]
                # Get integer part (lattice translation)
                dis_int = np.round(dis)
                # Get fractional remainder
                dis_rem = dis - dis_int
                # Convert to Cartesian for distance check
                dis_rem_cart = lattice.T @ dis_rem
                dis_norm = np.linalg.norm(dis_rem_cart)
                
                if dis_norm < 1e-3:
                    jatom = j
                    break
            
            if jatom is None:
                # Could not find transformed atom - this shouldn't happen
                # for valid symmetries, so skip this symmetry
                break
                
            # Check if atom types match
            if atom_types[iatom] != atom_types[jatom]:
                # Atom types don't match - invalid symmetry
                break
            
            # Check if magnetic moments match (with or without time-reversal)
            mag_diff_direct = np.linalg.norm(magmom[jatom] - mag_roted)
            mag_diff_reversed = np.linalg.norm(magmom[jatom] + mag_roted)
            
            if mag_diff_direct < abs(symm_magnetic_tolerance):
                # Magnetic moment preserved: S·m_i = m_j
                keep_symm_vote += 1
            elif mag_diff_reversed < abs(symm_magnetic_tolerance):
                # Magnetic moment flipped: S·m_i = -m_j (time-reversal)
                tr_vote += 1
        
        # Decide whether to keep this symmetry
        if tot_votes == 0:
            # No magnetic atoms found - keep all symmetries
            # (This is the non-magnetic case)
            filtered_rotations.append(rot)
            filtered_translations.append(trans)
            TR_flags.append(False)
        elif keep_symm_vote == tot_votes:
            # All magnetic moments preserved - keep without time-reversal
            filtered_rotations.append(rot)
            filtered_translations.append(trans)
            TR_flags.append(False)
        elif tr_vote == tot_votes:
            # All magnetic moments flipped - keep with time-reversal
            filtered_rotations.append(rot)
            filtered_translations.append(trans)
            TR_flags.append(True)
        # Otherwise: mixed behavior - discard this symmetry
    
    return filtered_rotations, filtered_translations, TR_flags


def find_symmetries_with_spglib(
    lattice: npt.NDArray[np.float64],
    atom_positions: npt.NDArray[np.float64],
    atom_types: List[str],
    num_atoms_each: List[int],
    magmom: Optional[npt.NDArray[np.float64]] = None,
    symprec: float = 1e-5,
    output_file: Optional[str] = None
) -> SymmetryData:
    """
    Find symmetry operations using spglib.
    
    Equivalent to the C function behavior in src/readinput.c where spglib
    is used to detect symmetries and write them to symmetries.dat.
    
    Parameters
    ----------
    lattice : np.ndarray
        3x3 lattice matrix (row vectors)
    atom_positions : np.ndarray
        Nx3 array of atomic positions in fractional coordinates
    atom_types : List[str]
        List of element symbols for each atom type
    num_atoms_each : List[int]
        Number of atoms for each type
    magmom : np.ndarray, optional
        Magnetic moments for each atom (Nx3)
    symprec : float, optional
        Symmetry precision for spglib (default: 1e-5)
    output_file : str, optional
        Output file name for symmetries (default: "symmetries.dat")
        
    Returns
    -------
    SymmetryData
        Symmetry operations with global time-reversal flag
        
    Raises
    ------
    ImportError
        If spglib is not available
    SymmetryError
        If symmetry detection fails
        
    Notes
    -----
    This function:
    1. Calls spglib to get symmetries of the crystal structure
    2. Converts rotation matrices from integer to float
    3. Handles magnetic materials (if magmom provided)
    4. Writes symmetries to output file
    5. Returns SymmetryData object
    """
    try:
        import spglib
    except ImportError:
        raise ImportError(
            "spglib is required for automatic symmetry detection. "
            "Install with: pip install spglib"
        )
    
    if output_file is None:
        output_file = "symmetries.dat"
    
    # Prepare atom types as integers for spglib
    atom_numbers = []
    for i, num_atoms in enumerate(num_atoms_each):
        atom_numbers.extend([i + 1] * num_atoms)
    atom_numbers = np.array(atom_numbers, dtype=int)
    
    # spglib expects column-major lattice (transpose)
    lattice_spg = lattice.T.copy()
    
    # Create spglib cell tuple: (lattice, positions, numbers)
    cell = (lattice_spg, atom_positions, atom_numbers)
    
    # Get symmetry operations from spglib
    # Use get_symmetry to get rotations and translations
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    
    if symmetry is None:
        raise SymmetryError("spglib failed to find symmetries")
    
    # Extract rotations and translations
    # spglib returns rotations as integer matrices
    rotations_int = symmetry['rotations']
    translations = symmetry['translations']
    nsymm = len(rotations_int)
    
    # Convert integer rotations to float
    rotations = [rot.astype(np.float64) for rot in rotations_int]
    translations_list = [trans.copy() for trans in translations]
    
    # Initialize time-reversal flags (default: no time-reversal per operation)
    TR_flags = [False] * nsymm
    global_trsymm = True
    
    # Handle magnetic materials
    # If magmom is provided, filter symmetries based on magnetic order
    if magmom is not None and len(magmom) > 0:
        # Convert magmom to numpy array if it's a list
        magmom_array = np.array(magmom) if not isinstance(magmom, np.ndarray) else magmom
        
        # Ensure magmom is 2D array (Nx3)
        # If already processed by apply_saxis_transformation, it should be Nx3
        # If not (e.g., test code passing 2D directly), accept it as-is
        if magmom_array.ndim == 1:
            # Fallback: if still 1D, assume z-component
            # (This shouldn't happen if apply_saxis_transformation was called)
            magmom_array = np.column_stack([
                np.zeros(len(magmom_array)),
                np.zeros(len(magmom_array)),
                magmom_array
            ])
        
        # Check if there are any non-zero magnetic moments
        has_magnetism = np.any(np.linalg.norm(magmom_array, axis=1) > 1e-6)
        
        if has_magnetism:
            # Apply magnetic symmetry filtering
            symm_magnetic_tolerance = 1e-3  # Default tolerance, could be parameter
            rotations, translations_list, TR_flags = derive_symm_for_magnetic_materials(
                rotations,
                translations_list,
                lattice,
                atom_positions,
                atom_numbers.tolist(),
                magmom_array,
                symm_magnetic_tolerance
            )
            nsymm = len(rotations)
            
            # If any symmetry has time-reversal, disable global time-reversal
            if any(TR_flags):
                global_trsymm = False
    
    # Create SymmetryOperation objects
    operations = []
    for i in range(nsymm):
        operations.append(SymmetryOperation(
            rotation=rotations[i],
            translation=translations_list[i],
            time_reversal=TR_flags[i]
        ))
    
    # Get space group information
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is not None:
            international_symbol = dataset.international
            international_number = dataset.number
            hall_symbol = dataset.hall
        else:
            international_symbol = "Unknown"
            international_number = 0
            hall_symbol = "Unknown"
    except:
        international_symbol = "Unknown"
        international_number = 0
        hall_symbol = "Unknown"
    
    # Write symmetries to file
    with open(output_file, 'w') as f:
        f.write("space group infomation:\n")
        f.write(f"    International: {international_symbol} ({international_number})\n")
        f.write(f"    schoenflies: {hall_symbol} ({international_number})\n")
        f.write(f"global time-reversal symmetry = {global_trsymm}\n")
        
        if magmom is not None:
            f.write("magnetic order detected\n")
            f.write("symmetries in the corresponding magnetic group:\n")
        
        f.write(f"nsymm = {nsymm}\n")
        
        for i, op in enumerate(operations):
            f.write(f"--- {i+1} ---\n")
            # Write rotation matrix as integers
            for j in range(3):
                f.write(f"{int(np.round(op.rotation[j, 0])):2d} "
                       f"{int(np.round(op.rotation[j, 1])):2d} "
                       f"{int(np.round(op.rotation[j, 2])):2d}\n")
            # Write translation and time-reversal flag
            f.write(f"{op.translation[0]:f} {op.translation[1]:f} {op.translation[2]:f}")
            if not global_trsymm:
                f.write(" T" if op.time_reversal else " F")
            f.write("\n")
    
    return SymmetryData(
        operations=operations,
        global_time_reversal=global_trsymm
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
