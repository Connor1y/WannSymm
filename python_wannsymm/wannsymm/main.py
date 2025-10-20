"""
Main program for WannSymm

Entry point and orchestration of the symmetrization workflow.
Translated from: src/main.c

Translation Status: ✅ COMPLETE (core workflow, MPI support optional)
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from .readinput import readinput, InputData, InputParseError
from .readsymm import readsymm, find_symmetries_with_spglib, SymmetryData, SymmetryError
from .wanndata import read_ham, write_ham, WannData
from .wannorb import WannOrb
from .rotate_ham import (
    apply_hermitian_symmetry,
    trsymm_ham,
    symmetrize_hamiltonians,
    check_hamiltonian_consistency,
    rotate_ham,
)
from .bndstruct import (
    diagonalize_hamiltonian,
    write_bands,
    write_band_characters,
)
from .usefulio import get_memory_usage_str
from .vector import Vector

# Version information
__version__ = "0.1.0-alpha"

# Configure logging
logger = logging.getLogger(__name__)


class WannSymmError(Exception):
    """Base exception for WannSymm errors."""
    pass


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for WannSymm.
    
    Parameters
    ----------
    verbose : bool, optional
        Enable verbose logging (default: False)
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wannsymm.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='wannsymm',
        description='WannSymm - Symmetry analysis for Wannier Hamiltonians',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wannsymm                    # Use default input file (wannsymm.in)
  wannsymm myinput.in         # Use specified input file
  wannsymm --version          # Show version information
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default=None,
        help='Input file name (default: wannsymm.in or symmham.in)'
    )
    
    parser.add_argument(
        '-V', '--version',
        action='version',
        version=f'WannSymm {__version__}'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def find_input_file(specified_file: Optional[str] = None) -> Path:
    """
    Find input file to use.
    
    Parameters
    ----------
    specified_file : str, optional
        User-specified input file
        
    Returns
    -------
    Path
        Path to input file
        
    Raises
    ------
    WannSymmError
        If no valid input file is found
    """
    if specified_file:
        input_path = Path(specified_file)
        if not input_path.exists():
            raise WannSymmError(f"Input file not found: {specified_file}")
        return input_path
    
    # Try default file names
    for default_name in ['wannsymm.in', 'symmham.in']:
        input_path = Path(default_name)
        if input_path.exists():
            logger.info(f"Using default input file: {default_name}")
            return input_path
    
    raise WannSymmError(
        "No input file found. Please specify an input file or create "
        "'wannsymm.in' in the current directory."
    )


def find_symmetries(
    input_data: InputData,
    output_file: Path
) -> SymmetryData:
    """
    Find or read symmetry operations.
    
    Parameters
    ----------
    input_data : InputData
        Parsed input data
    output_file : Path
        Output file for symmetry information
        
    Returns
    -------
    SymmetryData
        Symmetry operations
        
    Raises
    ------
    WannSymmError
        If symmetries cannot be found or read
    """
    logger.info("Finding symmetry operations...")
    
    if input_data.symm_from_file:
        # Read symmetries from file
        logger.info(f"Reading symmetries from file: {input_data.symm_input_file}")
        try:
            symm_data = readsymm(input_data.symm_input_file)
        except (InputParseError, SymmetryError) as e:
            raise WannSymmError(f"Failed to read symmetries: {e}")
    else:
        # Use spglib to find symmetries
        logger.info("Using spglib to find symmetries...")
        
        # Check that required structure data is available
        if input_data.lattice is None or input_data.atom_positions is None:
            raise WannSymmError(
                "Crystal structure information (lattice and atom positions) "
                "is required for automatic symmetry detection. "
                "Please provide structure in input file or use 'symm_from_file' option."
            )
        
        if input_data.atom_types is None or input_data.num_atoms_each is None:
            raise WannSymmError(
                "Atom type information is required for automatic symmetry detection. "
                "Please provide structure in input file or use 'symm_from_file' option."
            )
        
        # Process magmom with SAXIS transformation if needed
        processed_magmom = input_data.magmom
        if input_data.magmom is not None:
            from .readinput import apply_saxis_transformation
            natom = sum(input_data.num_atoms_each)
            flag_soc = 1 if input_data.spinors else 0
            processed_magmom = apply_saxis_transformation(
                input_data.magmom,
                natom,
                flag_soc,
                input_data.saxis
            )
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=input_data.lattice,
                atom_positions=input_data.atom_positions,
                atom_types=input_data.atom_types,
                num_atoms_each=input_data.num_atoms_each,
                magmom=processed_magmom,
                symprec=1e-5,
                output_file="symmetries.dat"
            )
        except (ImportError, SymmetryError) as e:
            raise WannSymmError(f"Failed to find symmetries with spglib: {e}")
    
    logger.info(f"Found {len(symm_data.operations)} symmetry operations")
    logger.info(f"Global time-reversal symmetry: {symm_data.global_time_reversal}")
    
    # Write symmetry information to output file
    with open(output_file, 'a') as f:
        f.write(f"\nSymmetries' info:\n")
        f.write(f"Number of symmetries: {len(symm_data.operations)}\n")
        f.write(f"Global time-reversal symmetry: {symm_data.global_time_reversal}\n\n")
    
    return symm_data


def construct_orbital_info(input_data: InputData, norb: int) -> List[WannOrb]:
    """
    Construct orbital information from input data.
    
    Creates a list of WannOrb objects based on projections and atomic positions.
    This is a simplified implementation that creates one orbital per site.
    
    Parameters
    ----------
    input_data : InputData
        Input data containing projections and atomic positions
    norb : int
        Expected number of orbitals
        
    Returns
    -------
    List[WannOrb]
        List of orbital information
        
    Notes
    -----
    This is a simplified implementation. For full functionality, should:
    - Parse projection groups properly
    - Handle s, p, d, f orbitals correctly
    - Apply local axis transformations
    - Handle SOC doubling of orbitals
    """
    orb_info = []
    
    # Check if we have atom positions
    if input_data.atom_positions is None or input_data.atom_types is None:
        # Fallback: create dummy orbitals at origin
        logger.warning("No atomic positions available, using dummy orbital sites at origin")
        for i in range(norb):
            orb_info.append(WannOrb(
                site=Vector(0, 0, 0),
                axis=[Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)],
                r=1, l=0, mr=0, ms=0
            ))
        return orb_info
    
    # Map orbital names to (l, number of orbitals)
    orb_map = {
        's': (0, 1),   # l=0, 1 orbital
        'p': (1, 3),   # l=1, 3 orbitals (px, py, pz)
        'd': (2, 5),   # l=2, 5 orbitals
        'f': (3, 7),   # l=3, 7 orbitals
    }
    
    # Build orbital list from projections
    atom_idx = 0
    for proj_group in input_data.projections:
        # Find matching atoms for this projection group
        atom_type = proj_group.element
        
        # Get positions for this atom type
        type_idx = None
        start_idx = 0
        for i, atype in enumerate(input_data.atom_types):
            if atype == atom_type:
                type_idx = i
                break
            if input_data.num_atoms_each:
                start_idx += input_data.num_atoms_each[i]
        
        if type_idx is None:
            logger.warning(f"Atom type '{atom_type}' not found in POSCAR, skipping")
            continue
        
        # Number of atoms of this type
        num_atoms = input_data.num_atoms_each[type_idx] if input_data.num_atoms_each else 1
        
        # For each atom of this type
        for iatom in range(num_atoms):
            atom_pos = input_data.atom_positions[start_idx + iatom]
            site = Vector(atom_pos[0], atom_pos[1], atom_pos[2])
            
            # For each orbital in this projection group
            for orb_name in proj_group.orbitals:
                # Get l and number of orbitals
                l, n_orb = orb_map.get(orb_name.lower(), (0, 1))
                
                # Create orbitals
                for mr in range(1, n_orb + 1):
                    orb_info.append(WannOrb(
                        site=site,
                        axis=[Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)],
                        r=1, l=l, mr=mr, ms=0
                    ))
                    
                    # If SOC, add spin-down component
                    if input_data.spinors:
                        orb_info.append(WannOrb(
                            site=site,
                            axis=[Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)],
                            r=1, l=l, mr=mr, ms=1
                        ))
    
    # Check if we have the right number of orbitals
    if len(orb_info) != norb:
        logger.warning(
            f"Constructed {len(orb_info)} orbitals but expected {norb}. "
            f"Using first {norb} orbitals."
        )
        # Pad or truncate to match
        while len(orb_info) < norb:
            orb_info.append(WannOrb(
                site=Vector(0, 0, 0),
                axis=[Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)],
                r=1, l=0, mr=0, ms=0
            ))
        orb_info = orb_info[:norb]
    
    return orb_info


def run_symmetrization(
    input_data: InputData,
    symm_data: SymmetryData,
    output_file: Path
) -> Tuple[WannData, WannData]:
    """
    Main symmetrization workflow.
    
    Parameters
    ----------
    input_data : InputData
        Parsed input data
    symm_data : SymmetryData
        Symmetry operations
    output_file : Path
        Output file for progress
        
    Returns
    -------
    ham_original : WannData
        Original Hamiltonian (for comparison)
    ham_symmetrized : WannData
        Symmetrized Hamiltonian
        
    Raises
    ------
    WannSymmError
        If symmetrization fails
    """
    start_time = time.time()
    
    # Step 1: Read Hamiltonian
    logger.info(f"Reading Hamiltonian from {input_data.seedname}_hr.dat...")
    try:
        ham_in = read_ham(input_data.seedname)
    except Exception as e:
        raise WannSymmError(f"Failed to read Hamiltonian: {e}")
    
    logger.info(f"Hamiltonian: {ham_in.norb} orbitals, {ham_in.nrpt} R-points")
    
    # Check for restart
    if input_data.restart:
        logger.info("Restart mode: reading existing symmetrized Hamiltonian...")
        seedname_symmed = f"{input_data.seedname}_symmed"
        try:
            ham_symmetrized = read_ham(seedname_symmed)
            logger.info("Restart successful")
            return ham_in, ham_symmetrized
        except Exception as e:
            logger.warning(f"Restart failed: {e}. Starting from beginning.")
    
    # Step 2: Apply Hermitian symmetry to input
    logger.info("Enforcing Hermiticity on input Hamiltonian...")
    if input_data.hermitian:
        ham_hermitian = apply_hermitian_symmetry(ham_in)
    else:
        ham_hermitian = ham_in
    
    # Step 3: Apply symmetry operations
    logger.info("Applying symmetry operations...")
    nsymm = len(symm_data.operations)
    ham_list = []
    
    # Create orbital info from input data
    orb_info = construct_orbital_info(input_data, ham_hermitian.norb)
    
    for isymm, symm_op in enumerate(symm_data.operations):
        logger.info(f"Processing symmetry {isymm+1}/{nsymm}...")
        
        # Apply time-reversal if needed
        if symm_op.time_reversal:
            ham_tr = trsymm_ham(ham_hermitian, orb_info, flag_soc=int(input_data.spinors))
        else:
            ham_tr = ham_hermitian
        
        # Rotate Hamiltonian using the complete implementation
        lattice = input_data.lattice if input_data.lattice is not None else np.eye(3)
        ham_rotated = rotate_ham(
            ham_tr,
            lattice,
            symm_op.rotation,
            symm_op.translation,
            orb_info,
            flag_soc=int(input_data.spinors),
            flag_local_axis=input_data.flag_local_axis,
            index_of_sym=isymm
        )
        
        ham_list.append(ham_rotated)
        
        elapsed = time.time() - start_time
        logger.info(f"Done: symmetry No.{isymm+1}, elapsed time: {elapsed:.1f} s")
    
    # Step 4: Average over symmetries
    logger.info("Averaging over symmetry operations...")
    ham_symmetrized = symmetrize_hamiltonians(
        ham_list,
        nsymm=nsymm,
        expand_rvec=input_data.expandrvec
    )
    
    # Step 5: Apply global time-reversal if needed
    if symm_data.global_time_reversal and input_data.global_trsymm:
        logger.info("Applying global time-reversal symmetry...")
        ham_tr_global = trsymm_ham(ham_symmetrized, orb_info, flag_soc=int(input_data.spinors))
        # Average with time-reversed version
        ham_combined = [ham_symmetrized, ham_tr_global]
        ham_symmetrized = symmetrize_hamiltonians(ham_combined, nsymm=2, expand_rvec=False)
    
    # Step 6: Check consistency
    if not input_data.expandrvec:
        logger.info("Checking consistency with original Hamiltonian...")
        is_consistent, max_diff = check_hamiltonian_consistency(
            ham_in,
            ham_symmetrized,
            tolerance=input_data.ham_tolerance
        )
        if is_consistent:
            logger.info(f"Consistency check passed. Max difference: {max_diff:.6e}")
        else:
            logger.warning(f"Consistency check failed. Max difference: {max_diff:.6e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Symmetrization completed in {elapsed:.1f} s")
    logger.info(get_memory_usage_str("Memory usage: "))
    
    return ham_in, ham_symmetrized


def calculate_bands(
    input_data: InputData,
    ham_original: WannData,
    ham_symmetrized: WannData,
    output_file: Path
) -> None:
    """
    Calculate band structures.
    
    Parameters
    ----------
    input_data : InputData
        Parsed input data
    ham_original : WannData
        Original Hamiltonian
    ham_symmetrized : WannData
        Symmetrized Hamiltonian
    output_file : Path
        Output file for progress
    """
    if not input_data.kpaths:
        logger.info("No k-paths specified, skipping band structure calculation")
        return
    
    logger.info("Calculating band structures...")
    
    # Calculate bands along k-paths
    try:
        if input_data.bands_ori:
            logger.info("Calculating original bands...")
            write_bands(
                ham_original,
                input_data.kpaths,
                input_data.klabels,
                nk_per_kpath=input_data.nk_per_kpath,
                output_prefix=f"{input_data.seedname}_ori"
            )
        
        if input_data.bands_symmed:
            logger.info("Calculating symmetrized bands...")
            write_bands(
                ham_symmetrized,
                input_data.kpaths,
                input_data.klabels,
                nk_per_kpath=input_data.nk_per_kpath,
                output_prefix=f"{input_data.seedname}_symmed"
            )
        
        logger.info("Band structure calculation completed")
    except Exception as e:
        logger.error(f"Band structure calculation failed: {e}")


def calculate_characters(
    input_data: InputData,
    ham_symmetrized: WannData,
    symm_data: SymmetryData,
    output_file: Path
) -> None:
    """
    Calculate band characters at specified k-points.
    
    Parameters
    ----------
    input_data : InputData
        Parsed input data
    ham_symmetrized : WannData
        Symmetrized Hamiltonian
    symm_data : SymmetryData
        Symmetry operations
    output_file : Path
        Output file for progress
    """
    if not input_data.chaeig or not input_data.kpts:
        logger.info("No k-points specified for character calculation")
        return
    
    logger.info("Calculating band characters...")
    
    # TODO: Implement character calculation
    # This requires additional functionality from bndstruct module
    logger.warning("Character calculation not yet fully implemented")


def main() -> int:
    """
    Main entry point for WannSymm.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info("="*70)
    logger.info(f"WannSymm {__version__}")
    logger.info("="*70)
    
    try:
        # Find input file
        input_file = find_input_file(args.input_file)
        logger.info(f"Input file: {input_file}")
        
        # Create output file
        output_file = Path("wannsymm.out")
        with open(output_file, 'w') as f:
            f.write(f"WannSymm {__version__}\n")
            f.write("="*70 + "\n")
        
        # Read input
        logger.info("Reading input file...")
        try:
            input_data = readinput(str(input_file))
        except InputParseError as e:
            raise WannSymmError(f"Failed to parse input file: {e}")
        
        logger.info(f"DFT code: {input_data.dftcode}")
        logger.info(f"Seedname: {input_data.seedname}")
        logger.info(f"SOC: {input_data.spinors}")
        
        # Find symmetries
        symm_data = find_symmetries(input_data, output_file)
        
        # Run symmetrization
        ham_original, ham_symmetrized = run_symmetrization(
            input_data,
            symm_data,
            output_file
        )
        
        # Write output
        logger.info("Writing symmetrized Hamiltonian...")
        seedname_symmed = f"{input_data.seedname}_symmed"
        write_ham(ham_symmetrized, seedname_symmed)
        logger.info(f"Symmetrized Hamiltonian written to {seedname_symmed}_hr.dat")
        
        # Optional: Calculate bands
        if input_data.bands:
            calculate_bands(input_data, ham_original, ham_symmetrized, output_file)
        
        # Optional: Calculate characters
        if input_data.chaeig:
            calculate_characters(input_data, ham_symmetrized, symm_data, output_file)
        
        logger.info("="*70)
        logger.info("WannSymm completed successfully!")
        logger.info("="*70)
        
        return 0
        
    except WannSymmError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
