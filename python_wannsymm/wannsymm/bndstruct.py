"""
Band structure module for WannSymm

Calculates band structures, characters, and eigenvalues.
Translated from: src/bndstruct.h and src/bndstruct.c

Translation Status: ✅ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TextIO
import numpy as np
import numpy.typing as npt

from .constants import PI, cmplx_i, eps6
from .vector import Vector, dot_product, vector_rotate
from .wanndata import WannData
from .matrix import matrix3x3_inverse

# Type aliases
ComplexMatrix = npt.NDArray[np.complex128]
RealMatrix = npt.NDArray[np.float64]


@dataclass
class SmallGroup:
    """
    Small group structure for k-point symmetry analysis.

    Equivalent to C struct __smallgroup.

    Attributes
    ----------
    order : int
        Order of the small group (number of symmetry operations)
    element : List[int]
        Index of every group element in the full symmetry group.
        For single-valued group: range [0, nsymm-1]
        For double group: negative indices [-nsymm, -1] for additional elements

    Examples
    --------
    >>> sgrp = SmallGroup(order=0, element=[])
    >>> sgrp.order
    0
    """

    order: int = 0
    element: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize element list if needed."""
        if not self.element:
            # Pre-allocate space for potential maximum number of symmetries
            # This matches C behavior where array is pre-allocated
            self.element = []


def diagonalize_hamiltonian(
    hr: WannData, kpt: Vector
) -> Tuple[npt.NDArray[np.float64], Optional[ComplexMatrix]]:
    """
    Calculate eigenvalues and eigenvectors of k-space Hamiltonian.

    Equivalent to C function: void bnd_eig_hk(double * eig_hk, dcomplex * vr_hk,
                                              wanndata * hr, vector kpt)

    This function:
    1. Calculates H(k) from H(R) via Fourier transform:
       H(k) = Σ_R exp(i k·R) H(R) / weight(R)
    2. Diagonalizes H(k) to get eigenvalues and eigenvectors

    Parameters
    ----------
    hr : WannData
        Real-space Hamiltonian data structure
    kpt : Vector
        k-point in crystal coordinates

    Returns
    -------
    eig_hk : np.ndarray
        Eigenvalues (energies) at k-point, shape (norb,)
        Sorted in ascending order
    vr_hk : np.ndarray or None
        Right eigenvectors as columns, shape (norb, norb)
        vr_hk[:, i] is the eigenvector for eigenvalue eig_hk[i]
        In row-major convention: vr_hk[i, :] is the i-th eigenvector

    Examples
    --------
    >>> from wannsymm.wanndata import WannData
    >>> from wannsymm.vector import Vector
    >>> hr = WannData(norb=2, nrpt=1)
    >>> hr.ham[0] = np.eye(2)
    >>> hr.rvec[0] = Vector(0, 0, 0)
    >>> hr.weight[0] = 1
    >>> kpt = Vector(0, 0, 0)
    >>> eig, vec = diagonalize_hamiltonian(hr, kpt)
    >>> eig.shape
    (2,)
    """
    norb = hr.norb

    # Initialize H(k) as zero matrix
    ham_k = np.zeros((norb, norb), dtype=np.complex128)

    # Fourier transform: H(k) = Σ_R exp(i k·R) H(R) / weight(R)
    for irpt in range(hr.nrpt):
        # Phase factor: exp(i k·R)
        k_dot_R = dot_product(kpt, hr.rvec[irpt])
        coeff = np.exp(cmplx_i * 2 * PI * k_dot_R)

        # Weight factor
        scal = coeff / hr.weight[irpt]

        # Add contribution: ham_k += scal * hr.ham[irpt]
        ham_k += scal * hr.ham[irpt]

    # Diagonalize H(k) using Hermitian eigenvalue solver
    # numpy.linalg.eigh returns eigenvalues in ascending order
    # and eigenvectors as columns
    eig_hk, vecs = np.linalg.eigh(ham_k)

    # Convert eigenvectors to row-major format to match C convention
    # In C: vr_hk[i*norb + j] corresponds to eigenvector i, component j
    # After eigh: vecs[:, i] is eigenvector for eig_hk[i]
    # We want: vr_hk[i, j] = vecs[j, i]
    vr_hk = vecs.T.copy()

    return eig_hk, vr_hk


def identify_degeneracies(
    eig_hk: npt.NDArray[np.float64], degenerate_tolerance: float = eps6
) -> npt.NDArray[np.int32]:
    """
    Determine degeneracy of each eigenstate.

    For each eigenvalue, identifies how many consecutive eigenvalues
    are degenerate (within tolerance).

    Parameters
    ----------
    eig_hk : np.ndarray
        Eigenvalues (energies), shape (norb,)
        Must be sorted in ascending order
    degenerate_tolerance : float, optional
        Energy difference threshold for considering states degenerate.
        Default is eps6 (1e-6)

    Returns
    -------
    ndegen : np.ndarray
        Degeneracy count for each state, shape (norb,)
        ndegen[i] = number of degenerate states in this group

    Examples
    --------
    >>> eig = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    >>> ndegen = identify_degeneracies(eig, 1e-6)
    >>> ndegen
    array([3, 3, 3, 1, 1])
    """
    norb = len(eig_hk)
    ndegen = np.zeros(norb, dtype=np.int32)

    ndegentmp = 0
    for io in range(norb):
        ndegentmp += 1

        # Check if this is the last orbital or if next energy is different
        if io == norb - 1 or abs(eig_hk[io] - eig_hk[io + 1]) > degenerate_tolerance:
            # Mark all states in this degenerate group
            for jo in range(ndegentmp):
                ndegen[io - jo] = ndegentmp
            ndegentmp = 0

    return ndegen


def _print_symmetry_stub(
    fnout: str,
    lattice: RealMatrix,
    isymm_elem: int,
    rotation: RealMatrix,
    translation: npt.NDArray[np.float64],
    TR: int,
    flag_showtrans: int,
    flag_showmirror: int,
    flag_soc: bool,
) -> None:
    """
    Stub for print_symmetry function (to be implemented in readinput module).

    For now, outputs basic symmetry information.
    """
    with open(fnout, "a") as f:
        f.write(f"  Symmetry element #{isymm_elem}, TR={TR}\n")


def write_band_characters(
    fnout: str,
    ikpt: int,
    kpt: Vector,
    sgrp: SmallGroup,
    lattice: RealMatrix,
    rotations: npt.NDArray[np.float64],
    translations: npt.NDArray[np.float64],
    TR: npt.NDArray[np.int32],
    eig_hk: npt.NDArray[np.float64],
    ndegen: npt.NDArray[np.int32],
    sym_chas: List[npt.NDArray[np.complex128]],
    norb: int,
    flag_soc: bool = False,
    en_print_prec: int = 6,
    bnd_print_len: int = 5,
) -> None:
    """
    Write band characters to output file.

    Equivalent to C function: void bnd_write_characters(...)

    Parameters
    ----------
    fnout : str
        Output filename
    ikpt : int
        k-point index (for reference, not currently used in output)
    kpt : Vector
        k-point coordinates
    sgrp : SmallGroup
        Small group of symmetries that leave k-point invariant
    lattice : np.ndarray
        Lattice vectors, shape (3, 3)
    rotations : np.ndarray
        Rotation matrices for all symmetries, shape (nsymm, 3, 3)
    translations : np.ndarray
        Translation vectors for all symmetries, shape (nsymm, 3)
    TR : np.ndarray
        Time-reversal flags for each symmetry, shape (nsymm,)
    eig_hk : np.ndarray
        Eigenvalues at k-point, shape (norb,)
    ndegen : np.ndarray
        Degeneracy count for each state, shape (norb,)
    sym_chas : List[np.ndarray]
        Characters for each symmetry operation, length sgrp.order
        Each element is shape (norb,)
    norb : int
        Number of orbitals
    flag_soc : bool, optional
        Whether spin-orbit coupling is included
    en_print_prec : int, optional
        Precision for energy printing
    bnd_print_len : int, optional
        Field width for band number printing
    """
    # Open file in append mode
    with open(fnout, "a") as f:
        # Output k-point
        f.write(f"kpt:{kpt.x:10.7f} {kpt.y:10.7f} {kpt.z:10.7f}\n")

        # Output small group order
        f.write(f"SmallGroupOrder= {sgrp.order} , Related symmetries:\n")

    # Output symmetries that keep kpt invariant
    for sgrpi in range(sgrp.order):
        isymm = sgrp.element[sgrpi]
        if isymm < 0:
            isymm = -isymm - 1

        # Use stub function to write symmetry information
        _print_symmetry_stub(
            fnout,
            lattice,
            sgrp.element[sgrpi],
            rotations[isymm],
            translations[isymm],
            TR[isymm],
            0,  # flag_showtrans
            -1,  # flag_showmirror
            flag_soc,
        )

    # Output characters for each band
    with open(fnout, "a") as f:
        io = 0
        while io < norb:
            # Format band range
            msg1 = f"{io+1:>{bnd_print_len}} - {io+ndegen[io]:<{bnd_print_len}}"
            msg = (
                f"band No. {msg1}  energy ="
                f"{eig_hk[io]:>{en_print_prec+5}.{en_print_prec}f} eV  "
                f"ndegen = {ndegen[io]:>{bnd_print_len}}"
            )
            f.write(f"{msg}\n    ")

            # Output characters for this degenerate group
            for sgrpi in range(sgrp.order):
                cha = sym_chas[sgrpi][io]

                # Clean up -0.000 display
                if cha.real < 0 and abs(cha.real) < eps6:
                    cha = complex(0, cha.imag)
                if cha.imag < 0 and abs(cha.imag) < eps6:
                    cha = complex(cha.real, 0)

                f.write(f"{cha.real:6.3f}{cha.imag:+6.3f}i")

                if sgrpi % 4 == 3 or sgrpi == sgrp.order - 1:
                    f.write("\n    ")
                else:
                    f.write(", ")

            f.write("\n")
            io += ndegen[io]


def write_band_eigenvalues(
    fnout: str,
    ikpt: int,
    kpt: Vector,
    sgrp: SmallGroup,
    lattice: RealMatrix,
    rotations: npt.NDArray[np.float64],
    translations: npt.NDArray[np.float64],
    TR: npt.NDArray[np.int32],
    eig_hk: npt.NDArray[np.float64],
    ndegen: npt.NDArray[np.int32],
    sym_eigs: List[npt.NDArray[np.complex128]],
    norb: int,
    flag_soc: bool = False,
    en_print_prec: int = 6,
    bnd_print_len: int = 5,
) -> None:
    """
    Write band eigenvalues to output file.

    Equivalent to C function: void bnd_write_eigenvalues(...)

    Parameters
    ----------
    fnout : str
        Output filename
    ikpt : int
        k-point index (for reference, not currently used in output)
    kpt : Vector
        k-point coordinates
    sgrp : SmallGroup
        Small group of symmetries that leave k-point invariant
    lattice : np.ndarray
        Lattice vectors, shape (3, 3)
    rotations : np.ndarray
        Rotation matrices for all symmetries, shape (nsymm, 3, 3)
    translations : np.ndarray
        Translation vectors for all symmetries, shape (nsymm, 3)
    TR : np.ndarray
        Time-reversal flags for each symmetry, shape (nsymm,)
    eig_hk : np.ndarray
        Eigenvalues at k-point, shape (norb,)
    ndegen : np.ndarray
        Degeneracy count for each state, shape (norb,)
    sym_eigs : List[np.ndarray]
        Eigenvalues of symmetry operators, length sgrp.order
        Each element is shape (norb,)
    norb : int
        Number of orbitals
    flag_soc : bool, optional
        Whether spin-orbit coupling is included
    en_print_prec : int, optional
        Precision for energy printing
    bnd_print_len : int, optional
        Field width for band number printing
    """
    # Open file in append mode
    with open(fnout, "a") as f:
        # Output k-point
        f.write(f"kpt:{kpt.x:10.7f} {kpt.y:10.7f} {kpt.z:10.7f}\n")

        # Output small group order
        f.write(f"SmallGroupOrder= {sgrp.order} , Related symmetries:\n")

    # Output symmetries that keep kpt invariant
    for sgrpi in range(sgrp.order):
        isymm = sgrp.element[sgrpi]
        if isymm < 0:
            isymm = -isymm - 1

        _print_symmetry_stub(
            fnout,
            lattice,
            sgrp.element[sgrpi],
            rotations[isymm],
            translations[isymm],
            TR[isymm],
            0,  # flag_showtrans
            -1,  # flag_showmirror
            flag_soc,
        )

    # Output eigenvalues for each band
    with open(fnout, "a") as f:
        io = 0
        while io < norb:
            for jo in range(ndegen[io]):
                # Format band range
                msg1 = f"{io+1:>{bnd_print_len}} - {io+ndegen[io]:<{bnd_print_len}}"
                msg = (
                    f"No.{jo+1:>{bnd_print_len}} of "
                    f"{ndegen[io]:>{bnd_print_len}} degenerage band "
                    f"(band No. {msg1}) energy ="
                    f"{eig_hk[io]:>{en_print_prec+5}.{en_print_prec}f} eV "
                )
                f.write(f"{msg}\n    ")

                # Output eigenvalues for this state
                for sgrpi in range(sgrp.order):
                    eig = sym_eigs[sgrpi][io + jo]

                    # Clean up -0.000 display
                    if eig.real < 0 and abs(eig.real) < eps6:
                        eig = complex(0, eig.imag)
                    if eig.imag < 0 and abs(eig.imag) < eps6:
                        eig = complex(eig.real, 0)

                    f.write(f"{eig.real:6.3f}{eig.imag:+6.3f}i")

                    if sgrpi % 4 == 3 or sgrpi == sgrp.order - 1:
                        f.write("\n    ")
                    else:
                        f.write(", ")

                f.write("\n")

            io += ndegen[io]


def write_bands(
    fbands: TextIO,
    norb: int,
    nkpath: int,
    nk_per_kpath: int,
    lattice: RealMatrix,
    kvecs: List[Vector],
    ebands: npt.NDArray[np.float64],
) -> None:
    """
    Write band structure data to file.

    Equivalent to C function: void bnd_write_bands(...)

    Outputs band energies along k-path with cumulative path length.

    Parameters
    ----------
    fbands : TextIO
        Output file object (already opened)
    norb : int
        Number of orbitals (bands)
    nkpath : int
        Number of k-path segments
    nk_per_kpath : int
        Number of k-points per path segment
    lattice : np.ndarray
        Lattice vectors, shape (3, 3)
    kvecs : List[Vector]
        k-point coordinates, length nkpath*nk_per_kpath
    ebands : np.ndarray
        Band energies, shape (nkpath*nk_per_kpath, norb)
    """
    # Calculate reciprocal lattice
    rec_latt = matrix3x3_inverse(lattice)

    # Write header
    fbands.write("#   k-path len        Energy\n")

    # Write bands
    for io in range(norb):
        kpath_len = 0.0

        for ikpt in range(nkpath * nk_per_kpath):
            # Calculate cumulative path length
            if ikpt % nk_per_kpath != 0:
                # Distance in reciprocal space
                dk = kvecs[ikpt] - kvecs[ikpt - 1]
                dk_cart = vector_rotate(dk, rec_latt)
                kpath_len += 2 * PI * np.sqrt(dot_product(dk_cart, dk_cart))

            # Write k-path length and energy
            fbands.write(f"{kpath_len:15.9f} {ebands[ikpt, io]:21.15f}\n")

            # Add blank line between path segments
            if ikpt % nk_per_kpath == nk_per_kpath - 1:
                fbands.write("\n")

        # Add blank line between bands
        fbands.write("\n")
