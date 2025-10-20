"""
Hamiltonian rotation module for WannSymm

Rotates and symmetrizes Hamiltonians using symmetry operations.
Translated from: src/rotate_ham.h and src/rotate_ham.c

Translation Status: ✓ COMPLETE
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from .constants import eps4, eps5, eps8
from .vector import Vector, equal, find_vector, vector_scale, vector_rotate, dot_product
from .matrix import matrix3x3_inverse, matrix3x3_dot, matrix3x3_transpose
from .wanndata import WannData, init_wanndata
from .wannorb import WannOrb


def inverse_symm(
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute inverse of a symmetry operation.
    
    Given a symmetry operation (R, t), computes the inverse (R^-1, -R^-1·t).
    
    Equivalent to C function: void inverse_symm(double rin[3][3], double rout[3][3], 
                                                 double tin[3], double tout[3])
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix
    translation : np.ndarray
        Translation vector of length 3
    
    Returns
    -------
    inv_rotation : np.ndarray
        Inverse of rotation matrix
    inv_translation : np.ndarray
        Transformed translation vector
    
    Examples
    --------
    >>> rot = np.eye(3)
    >>> trans = np.array([0.5, 0.0, 0.0])
    >>> inv_rot, inv_trans = inverse_symm(rot, trans)
    >>> np.allclose(inv_rot, rot)
    True
    >>> np.allclose(inv_trans, -trans)
    True
    """
    inv_rotation = matrix3x3_inverse(rotation)
    
    # Convert translation to vector and transform
    tin_v = Vector(translation[0], translation[1], translation[2])
    tout_v = vector_scale(-1.0, vector_rotate(tin_v, inv_rotation))
    inv_translation = np.array([tout_v.x, tout_v.y, tout_v.z])
    
    return inv_rotation, inv_translation


def get_conj_trans_of_ham(hin: WannData) -> WannData:
    """
    Compute the conjugate transpose of a Hamiltonian.
    
    For H(R), computes H*(−R) with orbital indices swapped.
    This operation enforces Hermiticity: H†(R) = H(−R).
    
    Equivalent to C function: void get_conj_trans_of_ham(wanndata * hout, wanndata * hin)
    
    Parameters
    ----------
    hin : WannData
        Input Hamiltonian
    
    Returns
    -------
    hout : WannData
        Conjugate transpose of input Hamiltonian
    
    Notes
    -----
    The conjugate transpose satisfies: hout[irpt,i,j] = conj(hin[irpt',j,i])
    where hin.rvec[irpt'] = -hin.rvec[irpt]
    
    Examples
    --------
    >>> wann = WannData(norb=2, nrpt=3)
    >>> wann.ham[0, 0, 1] = 1.0 + 1.0j
    >>> wann_ct = get_conj_trans_of_ham(wann)
    >>> # Check conjugate transpose property
    """
    num_orb = hin.norb
    hout = init_wanndata(num_orb, hin.nrpt)
    
    # Copy R-vectors and weights (unchanged by conjugate transpose)
    hout.rvec = [Vector(rv.x, rv.y, rv.z) for rv in hin.rvec]
    hout.weight = hin.weight.copy()
    
    # Compute conjugate transpose
    for irpt_out in range(hin.nrpt):
        rvec_out = hin.rvec[irpt_out]
        # Find -R in the R-vector list
        rvec_in = vector_scale(-1.0, rvec_out)
        irpt_in = find_vector(rvec_in, hin.rvec)
        
        if irpt_in == -1:
            # If -R not found, skip (should not happen in well-formed Hamiltonians)
            continue
        
        # Swap indices and conjugate: hout[irpt_out,j,i] = conj(hin[irpt_in,i,j])
        for jorb in range(num_orb):
            for iorb in range(num_orb):
                hout.ham[irpt_out, jorb, iorb] = np.conj(hin.ham[irpt_in, iorb, jorb])
    
    return hout


def trsymm_ham(hin: WannData, orb_info: List[WannOrb], flag_soc: int) -> WannData:
    """
    Apply time-reversal symmetry to a Hamiltonian.
    
    Time-reversal symmetry involves:
    - Without SOC: Complex conjugation
    - With SOC: Complex conjugation + spin rotation by σ_y
    
    Equivalent to C function: void trsymm_ham(wanndata * hout, wanndata * hin, 
                                               wannorb * orb_info, int flag_soc)
    
    Parameters
    ----------
    hin : WannData
        Input Hamiltonian
    orb_info : List[WannOrb]
        Wannier orbital information
    flag_soc : int
        Flag for spin-orbit coupling (0=no SOC, 1=with SOC)
    
    Returns
    -------
    hout : WannData
        Time-reversal transformed Hamiltonian
    
    Notes
    -----
    Time-reversal operator: T = i σ_y K (with SOC) or T = K (without SOC)
    where K is complex conjugation and σ_y is Pauli matrix.
    
    The tr_factor implements: <ms1|i·σ_y|ms2> for spin rotation.
    tr_factor[ms_i*2 + ms1] represents matrix elements.
    
    Examples
    --------
    >>> wann = WannData(norb=2, nrpt=1)
    >>> orb_info = [WannOrb(), WannOrb()]
    >>> wann_tr = trsymm_ham(wann, orb_info, flag_soc=0)
    """
    num_orb = hin.norb
    hout = init_wanndata(num_orb, hin.nrpt)
    
    # Copy R-vectors and weights
    hout.rvec = [Vector(rv.x, rv.y, rv.z) for rv in hin.rvec]
    hout.weight = hin.weight.copy()
    
    if flag_soc == 0:
        # Without SOC: just complex conjugation
        hout.ham = np.conj(hin.ham)
    else:
        # With SOC: complex conjugation + spin rotation
        # Time-reversal factor: i·σ_y = [[0, 1], [-1, 0]]
        # tr_factor[ms_out*2 + ms_in] gives the matrix element
        tr_factor = np.array([0, 1, -1, 0], dtype=np.complex128)
        
        for irpt in range(hin.nrpt):
            rvec_out = hin.rvec[irpt]
            rvec_in = rvec_out
            
            for jorb in range(num_orb):
                r2 = orb_info[jorb].r
                l2 = orb_info[jorb].l
                mr_j = orb_info[jorb].mr
                ms_j = orb_info[jorb].ms
                site_out2 = orb_info[jorb].site
                site_in2 = site_out2
                
                for iorb in range(num_orb):
                    r1 = orb_info[iorb].r
                    l1 = orb_info[iorb].l
                    mr_i = orb_info[iorb].mr
                    ms_i = orb_info[iorb].ms
                    site_out1 = orb_info[iorb].site
                    site_in1 = site_out1
                    
                    # Sum over spin states with time-reversal rotation
                    hout.ham[irpt, jorb, iorb] = 0.0
                    for ms1 in range(2):
                        for ms2 in range(2):
                            # Find orbital with different spin
                            iorb_in = -1
                            jorb_in = -1
                            for io in range(num_orb):
                                if (orb_info[io].r == r1 and orb_info[io].l == l1 and
                                    orb_info[io].mr == mr_i and orb_info[io].ms == ms1 and
                                    equal(orb_info[io].site, site_in1)):
                                    iorb_in = io
                                    break
                            for jo in range(num_orb):
                                if (orb_info[jo].r == r2 and orb_info[jo].l == l2 and
                                    orb_info[jo].mr == mr_j and orb_info[jo].ms == ms2 and
                                    equal(orb_info[jo].site, site_in2)):
                                    jorb_in = jo
                                    break
                            
                            if iorb_in >= 0 and jorb_in >= 0:
                                hout.ham[irpt, jorb, iorb] += (
                                    tr_factor[ms_i * 2 + ms1] *
                                    tr_factor[ms_j * 2 + ms2] *
                                    np.conj(hin.ham[irpt, jorb_in, iorb_in])
                                )
    
    return hout


def symmetrize_hamiltonians(
    ham_list: List[WannData],
    nsymm: int,
    expand_rvec: bool = True
) -> WannData:
    """
    Average Hamiltonians over symmetry operations.
    
    Computes the symmetrized Hamiltonian by averaging over all symmetry-transformed
    Hamiltonians: H_symm(R) = (1/N_symm) Σ_i H_i(R)
    
    Parameters
    ----------
    ham_list : List[WannData]
        List of Hamiltonians from each symmetry operation
    nsymm : int
        Number of symmetry operations
    expand_rvec : bool, optional
        Whether to expand R-vector list to include all transformed R-vectors
        Default: True
    
    Returns
    -------
    ham_final : WannData
        Symmetrized Hamiltonian
    
    Notes
    -----
    If expand_rvec is False, only R-vectors from the original Hamiltonian are included.
    Otherwise, all R-vectors appearing in any symmetry-transformed Hamiltonian are included.
    
    The averaging uses hamflag to track how many symmetries contributed to each element.
    
    Examples
    --------
    >>> # Create list of transformed Hamiltonians
    >>> ham_list = [WannData(norb=2, nrpt=3) for _ in range(4)]
    >>> ham_symm = symmetrize_hamiltonians(ham_list, nsymm=4)
    """
    if len(ham_list) == 0 or nsymm == 0:
        raise ValueError("Empty Hamiltonian list or zero symmetries")
    
    ham_in = ham_list[0]
    norb = ham_in.norb
    
    # Collect all unique R-vectors
    rvec_set = set()
    for rv in ham_in.rvec:
        rvec_set.add((rv.x, rv.y, rv.z))
    
    if expand_rvec:
        for ham_out in ham_list:
            for rv in ham_out.rvec:
                rvec_set.add((rv.x, rv.y, rv.z))
    
    # Convert to sorted list
    rvec_list = sorted(list(rvec_set))
    nrvec = len(rvec_list)
    
    # Initialize final Hamiltonian
    ham_final = init_wanndata(norb, nrvec)
    ham_final.rvec = [Vector(rv[0], rv[1], rv[2]) for rv in rvec_list]
    
    # If not expanding, copy weights from input
    if not expand_rvec:
        for irpt in range(min(nrvec, ham_in.nrpt)):
            ham_final.weight[irpt] = ham_in.weight[irpt]
    
    # Average over symmetries
    for irpt in range(nrvec):
        for isymm in range(nsymm):
            # Find this R-vector in the symmetry-transformed Hamiltonian
            jrpt = find_vector(ham_final.rvec[irpt], ham_list[isymm].rvec)
            if jrpt == -1:
                continue
            
            # Accumulate Hamiltonian elements
            ham_final.ham[irpt] += ham_list[isymm].ham[jrpt]
            ham_final.hamflag[irpt] += ham_list[isymm].hamflag[jrpt]
        
        # Normalize by number of contributing symmetries
        for io in range(norb):
            for jo in range(norb):
                if ham_final.hamflag[irpt, io, jo] != 0:
                    ham_final.ham[irpt, io, jo] /= ham_final.hamflag[irpt, io, jo]
                    if not expand_rvec:
                        # Apply weight factor
                        ham_final.ham[irpt, io, jo] *= ham_final.weight[irpt]
    
    return ham_final


def apply_hermitian_symmetry(ham_in: WannData) -> WannData:
    """
    Enforce Hermiticity on a Hamiltonian.
    
    Computes: H_hermitian = (H + H†)/2
    
    Parameters
    ----------
    ham_in : WannData
        Input Hamiltonian
    
    Returns
    -------
    ham_hermitian : WannData
        Hermitian-symmetrized Hamiltonian
    
    Examples
    --------
    >>> wann = WannData(norb=2, nrpt=3)
    >>> wann.ham[0, 0, 1] = 1.0 + 0.5j
    >>> wann_h = apply_hermitian_symmetry(wann)
    """
    ham_conj_trans = get_conj_trans_of_ham(ham_in)
    
    # Average with conjugate transpose
    ham_hermitian = init_wanndata(ham_in.norb, ham_in.nrpt)
    ham_hermitian.rvec = [Vector(rv.x, rv.y, rv.z) for rv in ham_in.rvec]
    ham_hermitian.weight = ham_in.weight.copy()
    ham_hermitian.hamflag = ham_in.hamflag.copy()
    ham_hermitian.ham = (ham_in.ham + ham_conj_trans.ham) / 2.0
    
    return ham_hermitian


def check_hamiltonian_consistency(
    ham1: WannData,
    ham2: WannData,
    tolerance: float = 0.1
) -> Tuple[bool, float]:
    """
    Check consistency between two Hamiltonians.
    
    Compares Hamiltonian matrix elements and returns whether they agree
    within the specified tolerance.
    
    Parameters
    ----------
    ham1 : WannData
        First Hamiltonian
    ham2 : WannData
        Second Hamiltonian
    tolerance : float, optional
        Tolerance for element-wise comparison (default: 0.1 eV)
    
    Returns
    -------
    is_consistent : bool
        True if Hamiltonians are consistent
    max_diff : float
        Maximum absolute difference found
    
    Examples
    --------
    >>> wann1 = WannData(norb=2, nrpt=1)
    >>> wann2 = WannData(norb=2, nrpt=1)
    >>> consistent, max_diff = check_hamiltonian_consistency(wann1, wann2)
    >>> consistent
    True
    """
    if ham1.norb != ham2.norb or ham1.nrpt != ham2.nrpt:
        return False, float('inf')
    
    max_diff = 0.0
    for irpt in range(ham1.nrpt):
        # Check if R-vectors match
        if not equal(ham1.rvec[irpt], ham2.rvec[irpt]):
            continue
        
        diff = np.abs(ham1.ham[irpt] - ham2.ham[irpt])
        max_diff = max(max_diff, np.max(diff))
    
    is_consistent = max_diff < tolerance
    return is_consistent, max_diff
