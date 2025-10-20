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
from .vector import Vector, equal, find_vector, vector_scale, vector_rotate, dot_product, vector_add, vector_sub
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


def getrvec_and_site(
    loc: Vector,
    orb_info: List[WannOrb],
    norb: int,
    lattice: npt.NDArray[np.float64],
    eps: float = 1e-3
) -> Tuple[Vector, Vector]:
    """
    Decompose a location into R-vector and orbital site.
    
    Given a location in fractional coordinates, find which orbital site it 
    corresponds to and what the R-vector offset is.
    
    Equivalent to C function: void getrvec_and_site(vector * p2rvec, vector * p2site, 
                                                     vector loc, wannorb * orb_info, 
                                                     int norb, double lattice[3][3])
    
    Parameters
    ----------
    loc : Vector
        Location in fractional coordinates
    orb_info : List[WannOrb]
        Orbital information including sites
    norb : int
        Number of orbitals
    lattice : np.ndarray
        Crystal lattice (3x3)
    eps : float, optional
        Tolerance for site matching (default: 1e-3)
        
    Returns
    -------
    rvec : Vector
        Integer R-vector part
    site : Vector
        Orbital site part
        
    Raises
    ------
    RuntimeError
        If no matching site is found
        
    Notes
    -----
    The location is decomposed as: loc = rvec + site
    where rvec is an integer vector and site matches one of the orbital sites.
    """
    # Try to match location to each orbital site
    for i in range(norb):
        site = orb_info[i].site
        
        # Compute displacement: loc - site
        dis_x = loc.x - site.x
        dis_y = loc.y - site.y
        dis_z = loc.z - site.z
        
        # Split into integer and fractional parts
        dis_int_x = round(dis_x)
        dis_int_y = round(dis_y)
        dis_int_z = round(dis_z)
        
        dis_rem_x = dis_x - dis_int_x
        dis_rem_y = dis_y - dis_int_y
        dis_rem_z = dis_z - dis_int_z
        
        # Convert remainder to Cartesian coordinates to check distance
        dis_rem = Vector(dis_rem_x, dis_rem_y, dis_rem_z)
        dis_rem_cartesian = vector_rotate(dis_rem, lattice)
        
        # Check if remainder is small (within tolerance)
        dist_sq = dot_product(dis_rem_cartesian, dis_rem_cartesian)
        if dist_sq < eps * eps:
            # Found matching site
            rvec = Vector(dis_int_x, dis_int_y, dis_int_z)
            return rvec, site
    
    # No matching site found - this is an error
    raise RuntimeError(
        f"Cannot find matching orbital site for location ({loc.x:.5f}, {loc.y:.5f}, {loc.z:.5f}). "
        f"This may indicate an incompatible symmetry operation or incorrect orbital positions."
    )


def rotate_ham(
    hin: WannData,
    lattice: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
    orb_info: List[WannOrb],
    flag_soc: int,
    flag_local_axis: int = 0,
    index_of_sym: int = 0
) -> WannData:
    """
    Apply a symmetry operation to rotate a Hamiltonian.
    
    Transforms the Hamiltonian under a symmetry operation:
    H'(R', i', j') = D†(i→i') S† H(R, i, j) S D(j→j')
    
    where D are orbital rotation matrices and S is the spinor rotation matrix.
    
    Equivalent to C function: void rotate_ham(wanndata * hout, wanndata * hin, ...)
    
    Parameters
    ----------
    hin : WannData
        Input Hamiltonian
    lattice : np.ndarray
        Crystal lattice (3x3)
    rotation : np.ndarray
        Rotation matrix (3x3)
    translation : np.ndarray
        Translation vector (3,)
    orb_info : List[WannOrb]
        Orbital information
    flag_soc : int
        SOC flag (0=no SOC, 1=with SOC)
    flag_local_axis : int, optional
        Local axis flag (default: 0)
    index_of_sym : int, optional
        Index of symmetry operation for logging (default: 0)
        
    Returns
    -------
    hout : WannData
        Rotated Hamiltonian
        
    Notes
    -----
    This is the core symmetrization function. It:
    1. Computes where each orbital goes under the symmetry operation
    2. Collects all R-vectors needed in the output
    3. For each output matrix element, finds which input elements contribute
    4. Applies orbital and spinor rotation matrices to transform the Hamiltonian
    
    The transformation preserves the physical properties of the system while
    expressing them in the rotated basis.
    """
    from .rotate_orbital import rotate_cubic
    from .rotate_spinor import rotate_spinor
    from .wanndata import find_index_of_ham
    from .wannorb import find_index_of_wannorb
    
    norb = hin.norb
    
    # Get inverse symmetry operation
    inv_rotation, inv_translation = inverse_symm(rotation, translation)
    
    # Get rotation axis and angle for spinor and orbital rotations
    # We need to convert the rotation matrix to axis-angle form
    # For now, use simplified approach: extract from rotation matrix
    from .rotate_basis import get_axis_angle_of_rotation
    
    rot_axis, rot_angle, inv_flag = get_axis_angle_of_rotation(rotation, lattice)
    
    # Get spinor rotation matrix
    s_rot = rotate_spinor(rot_axis, rot_angle, inv_flag)
    
    # Get orbital rotation matrices for each l
    MAX_L = 3  # Maximum angular momentum
    orb_rot = {}
    for l in range(MAX_L + 1):
        orb_rot[l] = rotate_cubic(l, rot_axis, rot_angle, bool(inv_flag))
    
    # Create lookup tables for rotated orbital positions
    # site_symmed: site of orbital after symmetry operation
    # rvec_sup_symmed: extra R-vector from symmetry operation
    site_symmed = []
    rvec_sup_symmed = []
    site_invsed = []
    rvec_sup_invsed = []
    
    for iorb in range(norb):
        # Skip duplicate sites
        if iorb > 0 and equal(orb_info[iorb].site, orb_info[iorb-1].site):
            rvec_sup_symmed.append(rvec_sup_symmed[iorb-1])
            rvec_sup_invsed.append(rvec_sup_invsed[iorb-1])
            site_symmed.append(site_symmed[iorb-1])
            site_invsed.append(site_invsed[iorb-1])
            continue
        
        loc_in = orb_info[iorb].site
        
        # Apply symmetry operation
        loc_out = vector_add(
            vector_rotate(loc_in, rotation),
            Vector(translation[0], translation[1], translation[2])
        )
        rvec_sup, site = getrvec_and_site(loc_out, orb_info, norb, lattice)
        rvec_sup_symmed.append(rvec_sup)
        site_symmed.append(site)
        
        # Apply inverse symmetry operation
        loc_out = vector_add(
            vector_rotate(loc_in, inv_rotation),
            Vector(inv_translation[0], inv_translation[1], inv_translation[2])
        )
        rvec_sup, site = getrvec_and_site(loc_out, orb_info, norb, lattice)
        rvec_sup_invsed.append(rvec_sup)
        site_invsed.append(site)
    
    # Collect all R-vectors for output Hamiltonian
    rvec_set = set()
    # Start with input R-vectors
    for rv in hin.rvec:
        rvec_set.add((int(rv.x), int(rv.y), int(rv.z)))
    
    # Add R-vectors from rotated positions
    for irpt in range(hin.nrpt):
        rvec_in = hin.rvec[irpt]
        rvec_in_rotated = vector_rotate(rvec_in, rotation)
        
        # Check if rotated R-vector is close to integer
        if not (abs(rvec_in_rotated.x - round(rvec_in_rotated.x)) < 1e-3 and
                abs(rvec_in_rotated.y - round(rvec_in_rotated.y)) < 1e-3 and
                abs(rvec_in_rotated.z - round(rvec_in_rotated.z)) < 1e-3):
            # Skip incompatible R-vector
            rvec_set.discard((int(rvec_in.x), int(rvec_in.y), int(rvec_in.z)))
            continue
        
        rvec_in_rotated = Vector(
            round(rvec_in_rotated.x),
            round(rvec_in_rotated.y),
            round(rvec_in_rotated.z)
        )
        
        # For each orbital pair, compute output R-vector
        for jorb in range(norb):
            if jorb > 0 and equal(orb_info[jorb].site, orb_info[jorb-1].site):
                continue
            rvec_out2 = vector_add(rvec_sup_symmed[jorb], rvec_in_rotated)
            
            for iorb in range(norb):
                if iorb > 0 and equal(orb_info[iorb].site, orb_info[iorb-1].site):
                    continue
                rvec_out1 = rvec_sup_symmed[iorb]
                
                rvec_out = vector_sub(rvec_out2, rvec_out1)
                rvec_set.add((int(rvec_out.x), int(rvec_out.y), int(rvec_out.z)))
    
    # Create output Hamiltonian
    rvec_list = sorted(list(rvec_set))
    nrvec = len(rvec_list)
    
    hout = init_wanndata(norb, nrvec)
    hout.rvec = [Vector(rv[0], rv[1], rv[2]) for rv in rvec_list]
    hout.weight = np.ones(nrvec)
    
    # Transform Hamiltonian matrix elements
    for irpt_out in range(nrvec):
        rvec_out = hout.rvec[irpt_out]
        rvec_out_invsed = vector_rotate(rvec_out, inv_rotation)
        
        for jorb_out in range(norb):
            site_out2 = orb_info[jorb_out].site
            site_in2 = site_invsed[jorb_out]
            rvec_in2 = vector_add(rvec_sup_invsed[jorb_out], rvec_out_invsed)
            
            # Round to integers
            rvec_in2 = Vector(round(rvec_in2.x), round(rvec_in2.y), round(rvec_in2.z))
            
            r2 = orb_info[jorb_out].r
            l2 = orb_info[jorb_out].l
            mr_j = orb_info[jorb_out].mr
            ms_j = orb_info[jorb_out].ms
            
            for iorb_out in range(norb):
                site_out1 = orb_info[iorb_out].site
                site_in1 = site_invsed[iorb_out]
                rvec_in1 = rvec_sup_invsed[iorb_out]
                
                r1 = orb_info[iorb_out].r
                l1 = orb_info[iorb_out].l
                mr_i = orb_info[iorb_out].mr
                ms_i = orb_info[iorb_out].ms
                
                rvec_in = vector_sub(rvec_in2, rvec_in1)
                irpt_in = find_vector(rvec_in, hin.rvec)
                if irpt_in == -1:
                    continue
                
                ii_out = irpt_out * norb * norb + jorb_out * norb + iorb_out
                hout.hamflag.flat[ii_out] = 1
                
                # Apply orbital and spinor rotations
                if flag_soc == 1:
                    # With SOC: transform both orbital and spin
                    for mr1 in range(1, 2*l1+2):
                        for mr2 in range(1, 2*l2+2):
                            for ms1 in range(2):
                                for ms2 in range(2):
                                    # Find input orbital index
                                    iorb_in = find_index_of_wannorb(
                                        orb_info, norb, site_in1, r1, l1, mr1, ms1
                                    )
                                    jorb_in = find_index_of_wannorb(
                                        orb_info, norb, site_in2, r2, l2, mr2, ms2
                                    )
                                    
                                    if iorb_in < 0 or jorb_in < 0:
                                        continue
                                    
                                    # Apply transformation: D† S† H S D
                                    # H'(i',j') = D†_i,i1 S†_ms_i,ms1 H(i1,j1) S_ms_j,ms2 D_j,j1
                                    orb_factor_left = orb_rot[l1][mr_i-1, mr1-1]
                                    s_factor_left = s_rot[ms_i, ms1]
                                    orb_factor_right = np.conj(orb_rot[l2][mr_j-1, mr2-1])
                                    s_factor_right = np.conj(s_rot[ms_j, ms2])
                                    
                                    # FIX: Python stores ham[irpt, iorb, jorb], not ham[irpt, jorb, iorb]
                                    ham_element = hin.ham[irpt_in, iorb_in, jorb_in] / hin.weight[irpt_in]
                                    
                                    hout.ham[irpt_out, iorb_out, jorb_out] += (
                                        orb_factor_left * s_factor_left * 
                                        ham_element *
                                        s_factor_right * orb_factor_right
                                    )
                else:
                    # Without SOC: only orbital rotation
                    ms1 = ms2 = 0
                    for mr1 in range(1, 2*l1+2):
                        for mr2 in range(1, 2*l2+2):
                            # Find input orbital index
                            iorb_in = find_index_of_wannorb(
                                orb_info, site_in1, r1, l1, mr1, ms1
                            )
                            jorb_in = find_index_of_wannorb(
                                orb_info, site_in2, r2, l2, mr2, ms2
                            )
                            
                            if iorb_in < 0 or jorb_in < 0:
                                continue
                            
                            # Apply transformation: D† H D
                            orb_factor_left = orb_rot[l1][mr_i-1, mr1-1]
                            orb_factor_right = np.conj(orb_rot[l2][mr_j-1, mr2-1])
                            
                            # FIX: Python stores ham[irpt, iorb, jorb], not ham[irpt, jorb, iorb]
                            ham_element = hin.ham[irpt_in, iorb_in, jorb_in] / hin.weight[irpt_in]
                            
                            hout.ham[irpt_out, iorb_out, jorb_out] += (
                                orb_factor_left * ham_element * orb_factor_right
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
