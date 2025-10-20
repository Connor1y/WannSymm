"""
Wannier data module for WannSymm

Manages Hamiltonian data structures and I/O.
Translated from: src/wanndata.h and src/wanndata.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import numpy.typing as npt
import os

from .constants import eps8
from .vector import Vector, equal, find_vector
from .wannorb import WannOrb, find_index_of_wannorb


@dataclass
class WannData:
    """
    Wannier Hamiltonian data structure.

    Equivalent to C struct __wanndata.

    Stores Hamiltonian matrix elements in real space (R-space), indexed by
    R-vectors and orbital indices.

    Attributes
    ----------
    norb : int
        Number of Wannier orbitals
    nrpt : int
        Number of R-points (lattice vectors)
    ham : np.ndarray
        Complex Hamiltonian matrix elements, shape (nrpt, norb, norb).
        Stored in order: ham[irpt, iorb, jorb] = <0,jorb|H|R,iorb>
    hamflag : np.ndarray
        Integer flags indicating if ham[i] is enabled/meaningful.
        Shape: (nrpt, norb, norb). Value 1=enabled, 0=disabled
    rvec : List[Vector]
        List of R-vectors (lattice translation vectors)
    weight : np.ndarray
        Integer degeneracy weights for each R-point, shape (nrpt,)
    kvec : Vector
        k-vector (only used for Hamiltonian in k-space)

    Examples
    --------
    >>> wann = WannData(norb=2, nrpt=3)
    >>> wann.norb
    2
    >>> wann.ham.shape
    (3, 2, 2)
    """

    norb: int = 0
    nrpt: int = 0
    ham: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    hamflag: npt.NDArray[np.int32] = field(default_factory=lambda: np.array([]))
    rvec: List[Vector] = field(default_factory=list)
    weight: npt.NDArray[np.int32] = field(default_factory=lambda: np.array([]))
    kvec: Vector = field(default_factory=Vector)

    def __post_init__(self):
        """Initialize arrays after dataclass initialization."""
        if self.norb > 0 and self.nrpt > 0:
            # Initialize arrays if not already set
            if self.ham.size == 0:
                shape = (self.nrpt, self.norb, self.norb)
                self.ham = np.zeros(shape, dtype=np.complex128)
            if self.hamflag.size == 0:
                shape = (self.nrpt, self.norb, self.norb)
                self.hamflag = np.zeros(shape, dtype=np.int32)
            if len(self.rvec) == 0:
                self.rvec = [Vector() for _ in range(self.nrpt)]
            if self.weight.size == 0:
                self.weight = np.ones(self.nrpt, dtype=np.int32)


def init_wanndata(norb: int, nrpt: int) -> WannData:
    """
    Initialize a WannData structure.

    Equivalent to C function: void init_wanndata(wanndata * wann)

    Creates arrays for Hamiltonian elements, flags, R-vectors, and weights.
    All Hamiltonian elements are initialized to zero, flags to 0 (disabled),
    and weights to 1.

    Parameters
    ----------
    norb : int
        Number of Wannier orbitals
    nrpt : int
        Number of R-points

    Returns
    -------
    WannData
        Initialized WannData structure

    Examples
    --------
    >>> wann = init_wanndata(norb=4, nrpt=10)
    >>> wann.norb
    4
    >>> wann.nrpt
    10
    >>> wann.ham.shape
    (10, 4, 4)
    >>> np.all(wann.weight == 1)
    True
    """
    return WannData(norb=norb, nrpt=nrpt)


def read_ham(seed: str) -> WannData:
    """
    Read Hamiltonian from _hr.dat file.

    Equivalent to C function: void read_ham(wanndata * wann, char * seed)

    Reads Wannier90 formatted _hr.dat file containing the Hamiltonian
    matrix elements in real space.

    File format:
    - Line 1: Header comment
    - Line 2: Number of orbitals (norb)
    - Line 3: Number of R-points (nrpt)
    - Lines 4+: Degeneracy weights (15 per line)
    - Remaining lines: R-vector and matrix elements
      Format: rx ry rz iorb jorb real_part imag_part

    Parameters
    ----------
    seed : str
        Seed name for the file (e.g., "wannier90" for "wannier90_hr.dat")

    Returns
    -------
    WannData
        WannData structure with Hamiltonian data

    Raises
    ------
    FileNotFoundError
        If the _hr.dat file does not exist
    ValueError
        If file format is invalid

    Examples
    --------
    >>> # wann = read_ham("wannier90")
    >>> # wann.norb, wann.nrpt
    >>> # (12, 93)
    """
    filename = f"{seed}_hr.dat"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"ERROR!!! trying to read the file \"{filename}\", but it can not be found"
        )

    with open(filename, 'r') as fin:
        # Read header (line 1)
        fin.readline()  # Skip header

        # Read number of orbitals (line 2)
        line = fin.readline()
        norb = int(line.strip())

        # Read number of R-points (line 3)
        line = fin.readline()
        nrpt = int(line.strip())

        # Initialize WannData structure
        wann = init_wanndata(norb, nrpt)

        # Read degeneracy weights (15 per line)
        weights = []
        while len(weights) < nrpt:
            line = fin.readline()
            weights.extend([int(w) for w in line.split()])
        wann.weight = np.array(weights[:nrpt], dtype=np.int32)

        # Read Hamiltonian matrix elements
        for irpt in range(nrpt):
            for ielement in range(norb * norb):
                line = fin.readline()
                parts = line.split()

                t1, t2, t3 = int(parts[0]), int(parts[1]), int(parts[2])
                # Convert to 0-based indexing
                iorb_left = int(parts[3]) - 1   # jorb in C write
                iorb_right = int(parts[4]) - 1  # iorb in C write
                real_part = float(parts[5])
                imag_part = float(parts[6])

                # Store R-vector (only once per R-point)
                if iorb_right == 0 and iorb_left == 0:
                    wann.rvec[irpt] = Vector(t1, t2, t3)

                # Store Hamiltonian element
                # File format: columns 4,5 are (jorb+1, iorb+1)
                # C: ham[irpt*norb*norb + iorb*norb + jorb]
                # Python: ham[irpt, iorb, jorb]
                # Therefore: ham[irpt, iorb_right, iorb_left]
                val = real_part + 1j * imag_part
                wann.ham[irpt, iorb_right, iorb_left] = val
                wann.hamflag[irpt, iorb_right, iorb_left] = 1

    return wann


def write_ham(wann: WannData, seed: str) -> None:
    """
    Write Hamiltonian to _hr.dat file.

    Equivalent to C function: void write_ham(wanndata * wann, char * seed)

    Writes Hamiltonian in Wannier90 _hr.dat format.

    Parameters
    ----------
    wann : WannData
        WannData structure containing Hamiltonian
    seed : str
        Seed name for output file (e.g., "output" for "output_hr.dat")

    Examples
    --------
    >>> wann = init_wanndata(norb=2, nrpt=1)
    >>> wann.rvec[0] = Vector(0, 0, 0)
    >>> wann.ham[0, 0, 0] = 1.0 + 0.5j
    >>> # write_ham(wann, "test")
    """
    filename = f"{seed}_hr.dat"

    with open(filename, 'w') as fout:
        # Write header
        fout.write("# symmetrized Hamiltonian \n")

        # Write norb and nrpt
        fout.write(f"{wann.norb:5d}\n{wann.nrpt:5d}")

        # Write degeneracy weights (15 per line)
        for irpt in range(wann.nrpt):
            if irpt % 15 == 0:
                fout.write("\n")
            fout.write(f"{wann.weight[irpt]:5d}")
        fout.write("\n")

        # Write Hamiltonian matrix elements
        for irpt in range(wann.nrpt):
            for iorb in range(wann.norb):
                for jorb in range(wann.norb):
                    rx = int(wann.rvec[irpt].x)
                    ry = int(wann.rvec[irpt].y)
                    rz = int(wann.rvec[irpt].z)

                    real_part = np.real(wann.ham[irpt, iorb, jorb])
                    imag_part = np.imag(wann.ham[irpt, iorb, jorb])

                    fout.write(
                        f"{rx:5d}{ry:5d}{rz:5d}{jorb+1:5d}{iorb+1:5d}"
                        f"{real_part:22.16f}{imag_part:22.16f}\n"
                    )


def write_reduced_ham(wann: WannData, seed: str) -> None:
    """
    Write reduced Hamiltonian (only nonzero elements) to file.

    Equivalent to C function: void write_reduced_ham(wanndata * wann, char * seed)

    Writes only matrix elements with magnitude > eps8 to reduce file size.
    Also checks Hermiticity and warns if violated.

    Parameters
    ----------
    wann : WannData
        WannData structure containing Hamiltonian
    seed : str
        Seed name for output file

    Examples
    --------
    >>> wann = init_wanndata(norb=2, nrpt=1)
    >>> wann.rvec[0] = Vector(0, 0, 0)
    >>> wann.ham[0, 0, 0] = 1.0
    >>> # write_reduced_ham(wann, "test")
    """
    filename = f"{seed}_hr.dat"

    with open(filename, 'w') as fout:
        # Write header
        fout.write("# Reduced Wannier Hamiltonian (only nonzero value)\n")
        fout.write(f"{wann.norb:5d}\n{wann.nrpt:5d}")

        # Write degeneracy weights
        for irpt in range(wann.nrpt):
            if irpt % 15 == 0:
                fout.write("\n")
            fout.write(f"{wann.weight[irpt]:5d}")
        fout.write("\n")

        # Count nonzero elements
        nwann = 0
        for irpt in range(wann.nrpt):
            fout.write(
                f"{int(wann.rvec[irpt].x):5d}"
                f"{int(wann.rvec[irpt].y):5d}"
                f"{int(wann.rvec[irpt].z):5d}\n"
            )
            for iorb in range(wann.norb):
                for jorb in range(iorb + 1):
                    if np.abs(wann.ham[irpt, iorb, jorb]) > eps8:
                        nwann += 1

        fout.write(f"{nwann}\n")

        # Write nonzero elements and check Hermiticity
        for irpt in range(wann.nrpt):
            rv = Vector(-wann.rvec[irpt].x, -wann.rvec[irpt].y, -wann.rvec[irpt].z)
            iirpt = find_vector(rv, wann.rvec)

            for iorb in range(wann.norb):
                for jorb in range(iorb + 1):
                    if np.abs(wann.ham[irpt, iorb, jorb]) > eps8:
                        real_part = np.real(wann.ham[irpt, iorb, jorb])
                        imag_part = np.imag(wann.ham[irpt, iorb, jorb])

                        fout.write(
                            f"{iorb+1:5d}{jorb+1:5d}{irpt+1:5d}{iirpt+1:5d}"
                            f"{real_part:22.16f}{imag_part:22.16f}\n"
                        )

                        # Check Hermiticity: H(R,i,j) = conj(H(-R,j,i))
                        if iirpt >= 0:
                            real_diff = abs(
                                real_part - np.real(wann.ham[iirpt, jorb, iorb])
                            )
                            imag_diff = abs(
                                imag_part + np.imag(wann.ham[iirpt, jorb, iorb])
                            )

                            if real_diff > eps8 or imag_diff > eps8:
                                rx = int(wann.rvec[irpt].x)
                                ry = int(wann.rvec[irpt].y)
                                rz = int(wann.rvec[irpt].z)
                                fout.write(
                                    f"!!!!WARNING: Unhermitian Hamiltonian:"
                                    f"{iorb+1:5d}{jorb+1:5d}{rx:5d}{ry:5d}{rz:5d}\n"
                                )


def find_index_of_ham(
    wann: WannData,
    orb_info: List[WannOrb],
    irpt: int,
    site1: Vector,
    r1: int,
    l1: int,
    mr1: int,
    ms1: int,
    site2: Vector,
    r2: int,
    l2: int,
    mr2: int,
    ms2: int,
) -> int:
    """
    Find index of Hamiltonian matrix element.

    Equivalent to C function: int find_index_of_ham(...)

    Finds the linear index in the flattened Hamiltonian array for a specific
    matrix element identified by R-vector index and orbital quantum numbers.

    Parameters
    ----------
    wann : WannData
        WannData structure
    orb_info : List[WannOrb]
        List of Wannier orbital information
    irpt : int
        R-vector index
    site1 : Vector
        Site position for bra orbital
    r1, l1, mr1, ms1 : int
        Quantum numbers for bra orbital
    site2 : Vector
        Site position for ket orbital
    r2, l2, mr2, ms2 : int
        Quantum numbers for ket orbital

    Returns
    -------
    int
        Linear index in flattened array, or negative error code:
        -1: R-vector not found
        -2: Bra orbital not found
        -3: Ket orbital not found

    Examples
    --------
    >>> wann = init_wanndata(norb=2, nrpt=1)
    >>> site = Vector(0, 0, 0)
    >>> axis = [Vector(1,0,0), Vector(0,1,0), Vector(0,0,1)]
    >>> orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
    >>> orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=1)
    >>> orb_info = [orb1, orb2]
    >>> idx = find_index_of_ham(wann, orb_info, 0, site, 1, 0, 0, 0,
    ...                         site, 1, 0, 0, 1)
    >>> idx >= 0
    True
    """
    norb = wann.norb

    # Check R-vector index
    if irpt < 0:
        return -1

    # Find bra orbital index
    iorb = find_index_of_wannorb(orb_info, site1, r1, l1, mr1, ms1)
    if iorb < 0:
        return -2

    # Find ket orbital index
    jorb = find_index_of_wannorb(orb_info, site2, r2, l2, mr2, ms2)
    if jorb < 0:
        return -3

    # Return linear index: irpt*norb*norb + jorb*norb + iorb
    return irpt * norb * norb + jorb * norb + iorb


def combine_wanndata(wup: WannData, wdn: WannData) -> WannData:
    """
    Combine spin-up and spin-down Hamiltonians.

    Equivalent to C function: int combine_wanndata(wanndata * out,
                                                    wanndata * wup,
                                                    wanndata * wdn)

    Creates a combined Hamiltonian with double the number of orbitals,
    where the first half are spin-up and second half are spin-down.

    Parameters
    ----------
    wup : WannData
        Spin-up Hamiltonian
    wdn : WannData
        Spin-down Hamiltonian

    Returns
    -------
    WannData
        Combined Hamiltonian with norb = wup.norb + wdn.norb

    Raises
    ------
    ValueError
        If R-vectors don't match between spin channels

    Examples
    --------
    >>> wup = init_wanndata(norb=2, nrpt=1)
    >>> wdn = init_wanndata(norb=2, nrpt=1)
    >>> wup.rvec[0] = Vector(0, 0, 0)
    >>> wdn.rvec[0] = Vector(0, 0, 0)
    >>> wcomb = combine_wanndata(wup, wdn)
    >>> wcomb.norb
    4
    """
    # Check that both have same number of R-points
    if wup.nrpt != wdn.nrpt:
        raise ValueError(
            f"Spin-up and spin-down must have same nrpt: {wup.nrpt} != {wdn.nrpt}"
        )

    # Check that R-vectors match
    for i in range(wup.nrpt):
        if not equal(wup.rvec[i], wdn.rvec[i]):
            raise ValueError(
                f"R-vectors don't match at index {i}: "
                f"{wup.rvec[i]} != {wdn.rvec[i]}"
            )

    # Create combined structure
    norb_total = wup.norb + wdn.norb
    out = init_wanndata(norb=norb_total, nrpt=wup.nrpt)

    # Copy R-vectors and weights
    out.rvec = [Vector(v.x, v.y, v.z) for v in wup.rvec]
    out.weight = wup.weight.copy()

    # Copy Hamiltonian elements
    for irpt in range(wup.nrpt):
        # Copy spin-up block (top-left)
        out.ham[irpt, :wup.norb, :wup.norb] = wup.ham[irpt, :, :]
        out.hamflag[irpt, :wup.norb, :wup.norb] = wup.hamflag[irpt, :, :]

        # Copy spin-down block (bottom-right)
        out.ham[irpt, wup.norb:, wup.norb:] = wdn.ham[irpt, :, :]
        out.hamflag[irpt, wup.norb:, wup.norb:] = wdn.hamflag[irpt, :, :]

        # Off-diagonal blocks remain zero (no spin-flip terms)

    return out
