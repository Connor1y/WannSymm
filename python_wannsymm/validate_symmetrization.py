#!/usr/bin/env python3
"""
Validation script for symmetrization core

Demonstrates and validates the key functionalities of the rotate_ham module.
"""

import numpy as np
from wannsymm.rotate_ham import (
    inverse_symm,
    get_conj_trans_of_ham,
    trsymm_ham,
    symmetrize_hamiltonians,
    apply_hermitian_symmetry,
    check_hamiltonian_consistency
)
from wannsymm.wanndata import init_wanndata
from wannsymm.wannorb import WannOrb
from wannsymm.vector import Vector


def test_inverse_symmetry():
    """Test symmetry inversion"""
    print("\n" + "="*70)
    print("TEST 1: Symmetry Inversion")
    print("="*70)
    
    # 90 degree rotation around z-axis
    angle = np.pi / 2
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    translation = np.array([0.5, 0.0, 0.0])
    
    inv_rot, inv_trans = inverse_symm(rotation, translation)
    
    # Verify R^-1 * R = I
    product = inv_rot @ rotation
    identity_check = np.allclose(product, np.eye(3))
    
    print(f"Original rotation:\n{rotation}")
    print(f"Inverse rotation:\n{inv_rot}")
    print(f"R^-1 * R = I: {identity_check} ✓" if identity_check else f"R^-1 * R = I: {identity_check} ✗")
    
    return identity_check


def test_hermiticity():
    """Test Hermiticity enforcement"""
    print("\n" + "="*70)
    print("TEST 2: Hermiticity Enforcement")
    print("="*70)
    
    # Create non-Hermitian Hamiltonian
    wann = init_wanndata(2, 3)
    wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
    wann.ham[0] = np.array([[1.0, 2.0+1.0j], [3.0-0.5j, 2.0]])
    wann.ham[1] = np.array([[0.5, 1.0+1.0j], [0.2, 0.3]])
    wann.ham[2] = np.array([[0.6, 0.1], [1.5-2.0j, 0.4]])
    
    print("Original H(R=0) (non-Hermitian):")
    print(wann.ham[0])
    
    # Apply Hermitian symmetry
    wann_h = apply_hermitian_symmetry(wann)
    
    print("\nAfter Hermitian symmetry H(R=0):")
    print(wann_h.ham[0])
    
    # Check Hermiticity
    H_R0 = wann_h.ham[0]
    is_hermitian = np.allclose(H_R0, H_R0.conj().T)
    
    print(f"\nH = H†: {is_hermitian} ✓" if is_hermitian else f"H = H†: {is_hermitian} ✗")
    
    return is_hermitian


def test_time_reversal():
    """Test time-reversal symmetry"""
    print("\n" + "="*70)
    print("TEST 3: Time-Reversal Symmetry")
    print("="*70)
    
    # Create Hamiltonian with complex off-diagonal
    wann = init_wanndata(2, 1)
    wann.rvec = [Vector(0, 0, 0)]
    wann.ham[0] = np.array([
        [1.0, 2.0 - 1.0j],
        [2.0 + 1.0j, 3.0]
    ])
    
    print("Original H (complex off-diagonal):")
    print(wann.ham[0])
    
    # Create orbital info
    site = Vector(0, 0, 0)
    axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    orb_info = [
        WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0),
        WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0)
    ]
    
    # Apply time-reversal (without SOC)
    wann_tr = trsymm_ham(wann, orb_info, flag_soc=0)
    
    print("\nAfter time-reversal H:")
    print(wann_tr.ham[0])
    
    # Verify complex conjugation
    expected = np.conj(wann.ham[0])
    is_conjugated = np.allclose(wann_tr.ham[0], expected)
    
    print(f"\nH_TR = H*: {is_conjugated} ✓" if is_conjugated else f"H_TR = H*: {is_conjugated} ✗")
    
    return is_conjugated


def test_symmetrization():
    """Test symmetrization averaging"""
    print("\n" + "="*70)
    print("TEST 4: Symmetrization Averaging")
    print("="*70)
    
    # Create two different Hamiltonians
    wann1 = init_wanndata(2, 1)
    wann1.rvec = [Vector(0, 0, 0)]
    wann1.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
    wann1.hamflag[0] = np.ones((2, 2), dtype=np.int32)
    
    wann2 = init_wanndata(2, 1)
    wann2.rvec = [Vector(0, 0, 0)]
    wann2.ham[0] = np.array([[3.0, 1.5], [1.5, 4.0]])
    wann2.hamflag[0] = np.ones((2, 2), dtype=np.int32)
    
    print("H_1:")
    print(wann1.ham[0])
    print("\nH_2:")
    print(wann2.ham[0])
    
    # Symmetrize
    ham_list = [wann1, wann2]
    ham_symm = symmetrize_hamiltonians(ham_list, nsymm=2)
    
    print("\nSymmetrized H_avg = (H_1 + H_2)/2:")
    print(ham_symm.ham[0])
    
    # Verify averaging
    expected = (wann1.ham[0] + wann2.ham[0]) / 2.0
    is_averaged = np.allclose(ham_symm.ham[0], expected)
    
    print(f"\nCorrect averaging: {is_averaged} ✓" if is_averaged else f"Correct averaging: {is_averaged} ✗")
    
    return is_averaged


def test_consistency():
    """Test consistency checking"""
    print("\n" + "="*70)
    print("TEST 5: Consistency Checking")
    print("="*70)
    
    wann1 = init_wanndata(2, 1)
    wann1.rvec = [Vector(0, 0, 0)]
    wann1.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
    
    wann2 = init_wanndata(2, 1)
    wann2.rvec = [Vector(0, 0, 0)]
    wann2.ham[0] = np.array([[1.02, 0.51], [0.51, 2.03]])
    
    print("H_1:")
    print(wann1.ham[0])
    print("\nH_2 (slightly different):")
    print(wann2.ham[0])
    
    # Check with tolerance
    tolerance = 0.1
    consistent, max_diff = check_hamiltonian_consistency(
        wann1, wann2, tolerance=tolerance
    )
    
    print(f"\nTolerance: {tolerance} eV")
    print(f"Max difference: {max_diff:.6f} eV")
    print(f"Consistent: {consistent} ✓" if consistent else f"Consistent: {consistent} ✗")
    
    return consistent


def main():
    """Run all validation tests"""
    print("\n")
    print("█" * 70)
    print(" " * 15 + "SYMMETRIZATION CORE VALIDATION")
    print("█" * 70)
    
    results = []
    
    # Run tests
    results.append(("Symmetry Inversion", test_inverse_symmetry()))
    results.append(("Hermiticity Enforcement", test_hermiticity()))
    results.append(("Time-Reversal Symmetry", test_time_reversal()))
    results.append(("Symmetrization Averaging", test_symmetrization()))
    results.append(("Consistency Checking", test_consistency()))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("="*70)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!\n")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED\n")
        return 1


if __name__ == "__main__":
    exit(main())
