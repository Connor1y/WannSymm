"""
Input file reading module for WannSymm

Parses wannsymm.in input file and POSCAR structure files.
Translated from: src/readinput.h and src/readinput.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import numpy.typing as npt
from pathlib import Path

from .vector import Vector


def expand_vasp_notation(value_str: str) -> List[float]:
    """
    Expand VASP-style notation like "4*0.0" or "2*1.0 3*2.0".
    
    Parameters
    ----------
    value_str : str
        String with VASP notation (e.g., "4*0.0" or "1.0 2.0 3*0.0")
    
    Returns
    -------
    list
        Expanded list of floats
    
    Examples
    --------
    >>> expand_vasp_notation("4*0.0")
    [0.0, 0.0, 0.0, 0.0]
    >>> expand_vasp_notation("1.0 2*3.0")
    [1.0, 3.0, 3.0]
    """
    result = []
    parts = value_str.split()
    
    for part in parts:
        if '*' in part:
            # Parse N*value notation
            count_str, value_str = part.split('*', 1)
            try:
                count = int(count_str)
                value = float(value_str)
                result.extend([value] * count)
            except ValueError:
                raise ValueError(f"Invalid notation: {part}")
        else:
            # Simple float value
            try:
                result.append(float(part))
            except ValueError:
                raise ValueError(f"Invalid float: {part}")
    
    return result


class InputParseError(Exception):
    """Exception raised when parsing input file fails."""
    
    def __init__(self, message: str, line_number: Optional[int] = None, line: Optional[str] = None):
        self.line_number = line_number
        self.line = line
        if line_number is not None:
            message = f"Line {line_number}: {message}"
        if line is not None:
            message += f"\n  Line content: {line.strip()}"
        super().__init__(message)


@dataclass
class ProjectionGroup:
    """
    Projection group data.
    
    Equivalent to C struct __projgroup.
    """
    element: str  # Element name or site coordinates (e.g., "Mn" or "f=0.5,0.5,0.5")
    orbitals: List[str]  # List of orbital names (e.g., ["s", "d"] or ["px", "py", "pz"])
    zaxis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    xaxis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaxis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radial: int = 1
    zona: float = 1.0


@dataclass
class InputData:
    """
    Input data container with all tags and default values.
    
    Stores all input parameters parsed from wannsymm.in file.
    """
    # Required tags
    dftcode: str = "VASP"  # VASP or QE
    spinors: bool = False  # SOC flag
    seedname: str = ""
    
    # Structure input
    poscar_file: Optional[str] = None
    lattice: Optional[npt.NDArray[np.float64]] = None
    atom_positions: Optional[npt.NDArray[np.float64]] = None
    atom_types: Optional[List[str]] = None
    num_atoms_each: Optional[List[int]] = None
    
    # Projections
    projections: List[ProjectionGroup] = field(default_factory=list)
    flag_local_axis: int = 0
    
    # K-points
    kpts: List[Vector] = field(default_factory=list)
    kpaths: List[Tuple[Vector, Vector]] = field(default_factory=list)
    klabels: List[Tuple[str, str]] = field(default_factory=list)
    nk_per_kpath: int = 100
    
    # Calculation flags
    bands: bool = False
    bands_symmed: bool = True
    bands_ori: bool = True
    chaeig: bool = False
    chaeig_in_kpath: bool = False
    restart: bool = False
    
    # Symmetry options
    global_trsymm: bool = True
    expandrvec: bool = True  # C default is True (flag_expandrvec=1)
    everysymm: bool = False
    hermitian: bool = True  # C default is True (flag_hermitian=1)
    symm_from_file: bool = False
    symm_input_file: str = ""
    
    # Tolerances
    symm_magnetic_tolerance: float = 1e-3
    ham_tolerance: float = 0.1
    degenerate_tolerance: float = 1e-4
    
    # Magnetism
    magmom: Optional[List[float]] = None
    saxis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    # Other flags
    output_mem_usage: bool = False


def parseline(line: str, ignorecase: bool = True) -> Tuple[str, str]:
    """
    Parse a line from input file into tag and argument.
    
    Equivalent to C function: void parseline(char tag[], char arg[], 
                                             char inputline[], int ignorecase)
    
    Handles:
    - Comments starting with #, !, or //
    - Quoted strings
    - Tag-value separation by = or :
    - Case-insensitive tag names (optional)
    
    Parameters
    ----------
    line : str
        Input line to parse
    ignorecase : bool, optional
        If True, convert tags to lowercase
        
    Returns
    -------
    Tuple[str, str]
        (tag, argument) pair. Returns ("", "") for empty/comment lines.
        
    Examples
    --------
    >>> parseline("DFTcode = VASP")
    ('dftcode', 'VASP')
    >>> parseline("Spinors: T")
    ('spinors', 'T')
    >>> parseline("# This is a comment")
    ('', '')
    >>> parseline("SeedName='my_seed'")
    ('seedname', 'my_seed')
    """
    # Remove comments
    for comment_char in ['#', '!']:
        if comment_char in line:
            line = line[:line.index(comment_char)]
    if '//' in line:
        line = line[:line.index('//')]
    
    line = line.strip()
    if not line:
        return ("", "")
    
    # Find separator (= or :)
    separator = None
    in_quotes = False
    quote_char = None
    
    for i, char in enumerate(line):
        if char in ["'", '"']:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif not in_quotes and char in ['=', ':']:
            separator = i
            break
    
    if separator is None:
        return ("", "")
    
    # Extract tag and argument
    tag = line[:separator].strip()
    arg = line[separator+1:].strip()
    
    # Remove quotes from argument
    if arg:
        if (arg.startswith("'") and arg.endswith("'")) or \
           (arg.startswith('"') and arg.endswith('"')):
            arg = arg[1:-1]
    
    # Convert tag to lowercase if requested
    if ignorecase:
        tag = tag.lower()
    
    return (tag, arg)


def str2boolean(arg: str) -> Optional[bool]:
    """
    Convert string to boolean value.
    
    Equivalent to C function: int str2boolean(char * arg)
    
    Allowed values:
    - False: F, f, 0, .F, .f, .False., .FALSE.
    - True: T, t, 1, .T, .t, .True., .TRUE.
    
    Parameters
    ----------
    arg : str
        String to convert
        
    Returns
    -------
    Optional[bool]
        True, False, or None if invalid
        
    Examples
    --------
    >>> str2boolean("T")
    True
    >>> str2boolean(".False.")
    False
    >>> str2boolean("1")
    True
    >>> str2boolean("invalid")
    """
    arg = arg.strip()
    if not arg:
        return None
    
    # Check for False values
    if arg[0] in ['F', 'f', '0']:
        return False
    if arg.startswith('.') and len(arg) > 1 and arg[1] in ['F', 'f']:
        return False
    
    # Check for True values
    if arg[0] in ['T', 't', '1']:
        return True
    if arg.startswith('.') and len(arg) > 1 and arg[1] in ['T', 't']:
        return True
    
    return None


def read_poscar(filename: str) -> Tuple[npt.NDArray[np.float64], 
                                         npt.NDArray[np.float64],
                                         List[str], 
                                         List[int]]:
    """
    Read structure from POSCAR file.
    
    Uses ASE (Atomic Simulation Environment) for robust POSCAR parsing.
    Supports both Direct and Cartesian coordinates.
    
    Parameters
    ----------
    filename : str
        Path to POSCAR file
        
    Returns
    -------
    Tuple containing:
        lattice : np.ndarray
            3x3 lattice matrix (Angstrom)
        positions : np.ndarray
            Nx3 atomic positions in fractional coordinates
        atom_types : List[str]
            List of element symbols for each atom type
        num_atoms_each : List[int]
            Number of atoms for each type
            
    Raises
    ------
    InputParseError
        If POSCAR file cannot be read or parsed
        
    Examples
    --------
    >>> lattice, positions, types, nums = read_poscar("POSCAR")
    >>> lattice.shape
    (3, 3)
    >>> len(types) == len(nums)
    True
    """
    try:
        from ase.io import read
    except ImportError:
        raise ImportError(
            "ASE is required for POSCAR reading. "
            "Install with: pip install ase"
        )
    
    try:
        atoms = read(filename, format='vasp')
    except Exception as e:
        raise InputParseError(f"Failed to read POSCAR file '{filename}': {e}")
    
    # Get lattice vectors (cell)
    lattice = atoms.get_cell().array
    
    # Get scaled (fractional) positions
    positions = atoms.get_scaled_positions()
    
    # Get chemical symbols
    symbols = atoms.get_chemical_symbols()
    
    # Count atoms of each type
    atom_types = []
    num_atoms_each = []
    current_type = None
    
    for symbol in symbols:
        if symbol != current_type:
            atom_types.append(symbol)
            num_atoms_each.append(1)
            current_type = symbol
        else:
            num_atoms_each[-1] += 1
    
    return lattice, positions, atom_types, num_atoms_each


def parse_poscar_block(lines: List[str], 
                       start_line: int = 0) -> Tuple[npt.NDArray[np.float64],
                                                      npt.NDArray[np.float64],
                                                      List[str],
                                                      List[int],
                                                      int]:
    """
    Parse POSCAR format from lines in input file.
    
    Handles "begin structure_in_format_of_POSCAR" ... "end" blocks.
    
    Parameters
    ----------
    lines : List[str]
        List of input file lines
    start_line : int, optional
        Starting line number for error messages
        
    Returns
    -------
    Tuple containing:
        lattice : np.ndarray
            3x3 lattice matrix
        positions : np.ndarray
            Nx3 atomic positions (fractional)
        atom_types : List[str]
            Element symbols
        num_atoms_each : List[int]
            Number of atoms per type
        lines_consumed : int
            Number of lines parsed
    """
    import tempfile
    
    # Find end tag
    end_idx = None
    for i, line in enumerate(lines):
        tag, _ = parseline(line, ignorecase=True)
        if tag == 'end':
            end_idx = i
            break
    
    if end_idx is None:
        raise InputParseError(
            "Missing 'end' tag for structure_in_format_of_POSCAR block",
            line_number=start_line
        )
    
    # Write POSCAR content to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.poscar', delete=False) as f:
        temp_file = f.name
        # Add a comment line (required by POSCAR format)
        f.write("Structure from input file\n")
        for line in lines[:end_idx]:
            f.write(line + '\n')
    
    try:
        lattice, positions, atom_types, num_atoms_each = read_poscar(temp_file)
    finally:
        # Clean up temporary file
        Path(temp_file).unlink(missing_ok=True)
    
    return lattice, positions, atom_types, num_atoms_each, end_idx + 1


def parse_projections_block(lines: List[str],
                            start_line: int = 0) -> Tuple[List[ProjectionGroup], 
                                                           int, 
                                                           int]:
    """
    Parse projections block from input file.
    
    Handles "begin projections" ... "end projections" blocks.
    
    Parameters
    ----------
    lines : List[str]
        List of input file lines
    start_line : int, optional
        Starting line number for error messages
        
    Returns
    -------
    Tuple containing:
        projections : List[ProjectionGroup]
            Parsed projection groups
        flag_local_axis : int
            Count of local axis specifications
        lines_consumed : int
            Number of lines parsed
            
    Examples
    --------
    >>> lines = ["V: d", "O: p", "end projections"]
    >>> projs, flag, n = parse_projections_block(lines)
    >>> len(projs)
    2
    >>> projs[0].element
    'V'
    >>> projs[0].orbitals
    ['d']
    """
    projections = []
    flag_local_axis = 0
    
    for i, line in enumerate(lines):
        # Check for end tag first (before parseline, as it has no = or :)
        line_lower = line.strip().lower()
        if line_lower.startswith('end'):
            return projections, flag_local_axis, i + 1
        
        tag, arg = parseline(line, ignorecase=False)
        
        if not tag:
            continue
        
        # Parse projection line: element: orbitals [: options]
        pg = ProjectionGroup(element=tag, orbitals=[])
        
        # Handle coordinate-based sites (f= or c=)
        if tag.startswith('f') or tag.startswith('c'):
            # Check if it's a coordinate specification
            if '=' in tag:
                pg.element = tag  # Keep as is
                arg = arg  # Orbitals are in arg
        
        # Parse orbitals and optional parameters
        parts = arg.split(':')
        
        # First part contains orbitals
        if parts:
            orb_str = parts[0].strip()
            # Split by , or ;
            for orb in orb_str.replace(';', ',').split(','):
                orb = orb.strip()
                if orb:
                    pg.orbitals.append(orb)
        
        # Parse optional parameters (x, y, z, r, zona)
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            
            # Parse key=value or key value
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
            elif ' ' in part:
                key, value = part.split(None, 1)
                key = key.strip().lower()
                value = value.strip()
            else:
                continue
            
            # Parse based on key
            if key == 'x':
                pg.xaxis = tuple(map(float, value.split(',')))
                flag_local_axis += 1
            elif key == 'y':
                pg.yaxis = tuple(map(float, value.split(',')))
                flag_local_axis += 1
            elif key == 'z':
                pg.zaxis = tuple(map(float, value.split(',')))
                flag_local_axis += 1
            elif key == 'r':
                pg.radial = int(value)
            elif key == 'zona':
                pg.zona = float(value)
        
        if pg.orbitals:
            projections.append(pg)
    
    raise InputParseError(
        "Missing 'end' tag for projections block",
        line_number=start_line
    )


def parse_kpath_block(lines: List[str],
                     start_line: int = 0) -> Tuple[List[Tuple[Vector, Vector]],
                                                    List[Tuple[str, str]],
                                                    int]:
    """
    Parse k-path block from input file.
    
    Handles "begin kpath" ... "end" blocks.
    
    Format: label1 kx1 ky1 kz1  label2 kx2 ky2 kz2
    
    Parameters
    ----------
    lines : List[str]
        List of input file lines
    start_line : int, optional
        Starting line number for error messages
        
    Returns
    -------
    Tuple containing:
        kpaths : List[Tuple[Vector, Vector]]
            List of k-point paths (start, end)
        klabels : List[Tuple[str, str]]
            List of label pairs for each path
        lines_consumed : int
            Number of lines parsed
            
    Examples
    --------
    >>> lines = ["G 0 0 0 X 0.5 0 0", "X 0.5 0 0 M 0.5 0.5 0", "end"]
    >>> paths, labels, n = parse_kpath_block(lines)
    >>> len(paths)
    2
    >>> labels[0]
    ('G', 'X')
    """
    kpaths = []
    klabels = []
    
    for i, line in enumerate(lines):
        # Check for end tag first (before cleaning, as it has no = or :)
        line_lower = line.strip().lower()
        if line_lower.startswith('end'):
            return kpaths, klabels, i + 1
        
        # Remove comments from line before parsing
        for comment_char in ['#', '!']:
            if comment_char in line:
                line = line[:line.index(comment_char)]
        if '//' in line:
            line = line[:line.index('//')]
        
        line = line.strip()
        
        if not line:
            continue
        
        # Parse line: label1 x1 y1 z1 label2 x2 y2 z2
        parts = line.split()
        if len(parts) < 8:
            continue
        
        try:
            label1 = parts[0]
            k1 = Vector(float(parts[1]), float(parts[2]), float(parts[3]))
            label2 = parts[4]
            k2 = Vector(float(parts[5]), float(parts[6]), float(parts[7]))
            
            kpaths.append((k1, k2))
            klabels.append((label1, label2))
        except (ValueError, IndexError) as e:
            raise InputParseError(
                f"Invalid k-path specification: {e}",
                line_number=start_line + i,
                line=line
            )
    
    raise InputParseError(
        "Missing 'end' tag for kpath block",
        line_number=start_line
    )


def parse_kpts_block(lines: List[str],
                    start_line: int = 0) -> Tuple[List[Vector], int]:
    """
    Parse k-points block from input file.
    
    Handles "begin kpts" ... "end" blocks.
    
    Parameters
    ----------
    lines : List[str]
        List of input file lines
    start_line : int, optional
        Starting line number for error messages
        
    Returns
    -------
    Tuple containing:
        kpts : List[Vector]
            List of k-points
        lines_consumed : int
            Number of lines parsed
    """
    kpts = []
    
    for i, line in enumerate(lines):
        # Check for end tag first (before cleaning, as it has no = or :)
        line_lower = line.strip().lower()
        if line_lower.startswith('end'):
            return kpts, i + 1
        
        # Remove comments from line before parsing
        for comment_char in ['#', '!']:
            if comment_char in line:
                line = line[:line.index(comment_char)]
        if '//' in line:
            line = line[:line.index('//')]
        
        line = line.strip()
        
        if not line:
            continue
        
        # Parse line: kx ky kz
        parts = line.split()
        if len(parts) >= 3:
            try:
                kpt = Vector(float(parts[0]), float(parts[1]), float(parts[2]))
                kpts.append(kpt)
            except (ValueError, IndexError) as e:
                raise InputParseError(
                    f"Invalid k-point specification: {e}",
                    line_number=start_line + i,
                    line=line
                )
    
    raise InputParseError(
        "Missing 'end' tag for kpts block",
        line_number=start_line
    )


def readinput(filename: str) -> InputData:
    """
    Read and parse wannsymm input file.
    
    Main input reading function. Equivalent to C function readinput().
    
    Parameters
    ----------
    filename : str
        Path to input file (typically "wannsymm.in")
        
    Returns
    -------
    InputData
        Parsed input data with all tags
        
    Raises
    ------
    InputParseError
        If input file is invalid or cannot be parsed
    FileNotFoundError
        If input file does not exist
        
    Examples
    --------
    >>> input_data = readinput("wannsymm.in")
    >>> input_data.dftcode
    'VASP'
    >>> input_data.spinors
    False
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file '{filename}' not found")
    
    # Read all lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Initialize input data with defaults
    data = InputData()
    
    # Parse line by line
    i = 0
    while i < len(lines):
        line = lines[i]
        line_number = i + 1
        
        # Check for "begin" blocks first (these don't have = or :)
        line_lower = line.strip().lower().replace(' ', '')
        
        # Handle projections block
        if line_lower.startswith('beginprojection'):
            projections, flag_local, n_lines = \
                parse_projections_block(lines[i+1:], start_line=line_number+1)
            data.projections = projections
            data.flag_local_axis = flag_local
            i += n_lines + 1
            continue
        
        # Handle k-path block  
        if line_lower.startswith('beginkpath') or line_lower.startswith('beginkpoint_path'):
            kpaths, klabels, n_lines = \
                parse_kpath_block(lines[i+1:], start_line=line_number+1)
            data.kpaths = kpaths
            data.klabels = klabels
            i += n_lines + 1
            continue
        
        # Handle k-points block
        if line_lower.startswith('beginkpt'):
            kpts, n_lines = parse_kpts_block(lines[i+1:], start_line=line_number+1)
            data.kpts = kpts
            i += n_lines + 1
            continue
        
        # Handle inline POSCAR
        if line_lower.startswith('beginstructure_in_format_of_poscar') or \
           line_lower == 'structure_in_format_of_poscar':
            lattice, positions, atom_types, num_atoms, n_lines = \
                parse_poscar_block(lines[i+1:], start_line=line_number+1)
            data.lattice = lattice
            data.atom_positions = positions
            data.atom_types = atom_types
            data.num_atoms_each = num_atoms
            i += n_lines + 1
            continue
        
        tag, arg = parseline(line, ignorecase=True)
        
        # Skip empty lines and comments
        if not tag:
            i += 1
            continue
        
        try:
            # Parse DFT code
            if tag == 'dftcode':
                if arg.upper() in ['VASP', '1']:
                    data.dftcode = 'VASP'
                elif arg.upper() in ['QE', '2']:
                    data.dftcode = 'QE'
                else:
                    raise InputParseError(
                        f"Unknown DFT code: {arg}. Must be VASP or QE",
                        line_number=line_number,
                        line=line
                    )
            
            # Parse spinors (SOC)
            elif tag == 'spinors':
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value for spinors: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.spinors = value
            
            # Parse seedname
            elif tag == 'seedname':
                data.seedname = arg
            
            # Parse POSCAR file
            elif tag in ['useposcar', 'use_poscar']:
                data.poscar_file = arg
                # Resolve path relative to input file directory
                poscar_path = Path(arg)
                if not poscar_path.is_absolute():
                    poscar_path = filepath.parent / poscar_path
                lattice, positions, atom_types, num_atoms = read_poscar(str(poscar_path))
                data.lattice = lattice
                data.atom_positions = positions
                data.atom_types = atom_types
                data.num_atoms_each = num_atoms
            
            # Parse single k-point
            if tag == 'kpt':
                parts = arg.split()
                if len(parts) >= 3:
                    kpt = Vector(float(parts[0]), float(parts[1]), float(parts[2]))
                    data.kpts.append(kpt)
            
            # Parse nk_per_kpath
            elif tag in ['nk_per_kpath', 'nkperkpath']:
                data.nk_per_kpath = max(2, int(arg))
            
            # Parse bands flags
            elif tag in ['bands', 'bands_plot', 'calculatebands', 'calculate_bands']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.bands_symmed = value
                data.bands_ori = value
                data.bands = value
            
            elif tag in ['bands_symmed', 'bands_symmed_plot', 'bands_symmetrized',
                        'bands_symmetrized_plot', 'calculatebandssymed',
                        'calculate_bands_symmed']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.bands_symmed = value
            
            elif tag in ['bands_ori', 'bands_ori_plot', 'bands_original',
                        'bands_original_plot', 'calculatebandsori',
                        'calculate_bands_ori']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.bands_ori = value
            
            # Parse character and eigenvalue flags
            elif tag in ['chaeig', 'calculate_chaeig', 'calculatechaeig',
                        'charactersandeigenvalues', 'calculatecharactersandeigenvalues',
                        'characters_and_eigenvalues', 'calculate_characters_and_eigenvalues']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.chaeig = value
            
            elif tag in ['characters_in_kpath', 'charactersinkpath',
                        'chaeig_in_kpath', 'chaeiginkpath',
                        'calculate_chaeig_in_kpath', 'calculatechaeiginkpath',
                        'calc_band_symm', 'calcbandsymm']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.chaeig_in_kpath = value
                if value:
                    data.chaeig = True
            
            # Parse restart flag
            elif tag == 'restart':
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.restart = value
            
            # Parse symmetry flags
            elif tag in ['trsymm', 'global_trsymm']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.global_trsymm = value
            
            elif tag == 'expandrvec':
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.expandrvec = value
            
            elif tag == 'everysymm':
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.everysymm = value
            
            elif tag in ['enforce_hermitian', 'enforcehermitian',
                        'enforcehermiticity', 'enforce_hermiticity']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.hermitian = value
            
            elif tag in ['usesymmetry', 'use_symmetry',
                        'symminputfile', 'symm_input_file']:
                data.symm_from_file = True
                data.symm_input_file = arg
            
            elif tag in ['outputmemusage', 'output_mem_usage']:
                value = str2boolean(arg)
                if value is None:
                    raise InputParseError(
                        f"Invalid boolean value: {arg}",
                        line_number=line_number,
                        line=line
                    )
                data.output_mem_usage = value
            
            # Parse tolerances
            elif tag in ['symmmagnetictolerance', 'symm_magnetic_tolerance']:
                data.symm_magnetic_tolerance = float(arg)
            
            elif tag in ['hamtolerance', 'ham_tolerance']:
                data.ham_tolerance = float(arg)
            
            elif tag in ['degeneratetolerance', 'degenerate_tolerance']:
                data.degenerate_tolerance = float(arg)
            
            elif tag in ['symmtolerance', 'symm_tolerance']:
                # Deprecated, map to ham_tolerance
                data.ham_tolerance = float(arg)
            
            # Parse magnetism
            elif tag == 'magmom':
                if arg.upper() != 'NULL':
                    # Parse magmom string with support for VASP notation (e.g., "4*0.0")
                    data.magmom = expand_vasp_notation(arg)
            
            elif tag == 'saxis':
                parts = arg.split()
                if len(parts) >= 3:
                    data.saxis = (float(parts[0]), float(parts[1]), float(parts[2]))
            
            # Unknown tag - just warn, don't error
            elif tag != 'null':
                pass  # Silently ignore unknown tags for compatibility
        
        except (ValueError, IndexError) as e:
            raise InputParseError(
                f"Error parsing tag '{tag}': {e}",
                line_number=line_number,
                line=line
            )
        
        i += 1
    
    # Post-processing: set bands flag based on bands_ori and bands_symmed
    # Only if bands is still the default value (not explicitly set)
    if data.bands_ori or data.bands_symmed:
        data.bands = True
    
    # If no k-paths, disable band calculations
    if not data.kpaths:
        if not data.bands:  # Only reset if bands=False
            data.bands_symmed = False
            data.bands_ori = False
        data.chaeig_in_kpath = False
    
    # If no k-points and no k-paths (or chaeig_in_kpath not set), disable chaeig  
    # But don't disable if explicitly set to True
    if not data.kpts and (not data.kpaths or not data.chaeig_in_kpath):
        if not data.chaeig:  # Only set if not explicitly enabled
            data.chaeig = False
    
    return data
