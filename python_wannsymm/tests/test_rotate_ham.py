"""
Tests for rotate_ham module

Test Hamiltonian rotation and symmetrization.
"""

import pytest
import numpy as np

from wannsymm.rotate_ham import (
    inverse_symm,
    get_conj_trans_of_ham,
    trsymm_ham,
    symmetrize_hamiltonians,
    apply_hermitian_symmetry,
    check_hamiltonian_consistency
)
from wannsymm.wanndata import WannData, init_wanndata
from wannsymm.wannorb import WannOrb
from wannsymm.vector import Vector, equal


class TestInverseSymm:
    """Test inverse_symm function"""
    
    def test_inverse_identity(self):
        """Test inverse of identity symmetry"""
        rot = np.eye(3)
        trans = np.zeros(3)
        inv_rot, inv_trans = inverse_symm(rot, trans)
        
        assert np.allclose(inv_rot, rot)
        assert np.allclose(inv_trans, trans)
    
    def test_inverse_translation(self):
        """Test inverse with translation"""
        rot = np.eye(3)
        trans = np.array([0.5, 0.0, 0.0])
        inv_rot, inv_trans = inverse_symm(rot, trans)
        
        assert np.allclose(inv_rot, rot)
        assert np.allclose(inv_trans, -trans)
    
    def test_inverse_rotation(self):
        """Test inverse with rotation"""
        # 90 degree rotation around z-axis
        angle = np.pi / 2
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        trans = np.zeros(3)
        inv_rot, inv_trans = inverse_symm(rot, trans)
        
        # Check R^-1 * R = I
        product = inv_rot @ rot
        assert np.allclose(product, np.eye(3))
    
    def test_inverse_combined(self):
        """Test inverse with rotation and translation"""
        angle = np.pi / 3
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        trans = np.array([0.5, 0.5, 0.0])
        inv_rot, inv_trans = inverse_symm(rot, trans)
        
        # Verify (R,t)^-1 * (R,t) = identity
        product_rot = inv_rot @ rot
        assert np.allclose(product_rot, np.eye(3))
        
        # Check that applying symmetry then inverse gives back original point
        # Proper way: (R^-1, t') where t' = -R^-1·t
        # To invert (R,t): x' = R·x + t, so x = R^-1·(x' - t) = R^-1·x' - R^-1·t
        point = np.array([1.0, 0.0, 0.0])
        transformed = rot @ point + trans
        # Apply inverse: x = R^-1·x' + t'
        recovered = inv_rot @ transformed + inv_trans
        assert np.allclose(recovered, point, atol=1e-10)


class TestGetConjTransOfHam:
    """Test get_conj_trans_of_ham function"""
    
    def test_conj_trans_identity(self):
        """Test conjugate transpose of identity Hamiltonian"""
        wann = init_wanndata(2, 3)
        wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        
        # Set up diagonal Hamiltonian (real)
        for irpt in range(3):
            for i in range(2):
                wann.ham[irpt, i, i] = 1.0
        
        wann_ct = get_conj_trans_of_ham(wann)
        
        # Check that diagonal elements are unchanged
        assert np.allclose(wann_ct.ham, wann.ham)
    
    def test_conj_trans_hermitian_property(self):
        """Test that conjugate transpose satisfies Hermiticity"""
        wann = init_wanndata(2, 3)
        wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        
        # Set up test Hamiltonian
        wann.ham[1, 0, 1] = 1.0 + 2.0j  # H(R)[0,1]
        wann.ham[2, 1, 0] = 3.0 - 4.0j  # H(-R)[1,0]
        
        wann_ct = get_conj_trans_of_ham(wann)
        
        # Check: hout[irpt(R), i, j] = conj(hin[irpt(-R), j, i])
        # For R=(1,0,0), irpt=1, -R=(-1,0,0), irpt=2
        assert np.allclose(wann_ct.ham[1, 0, 1], np.conj(wann.ham[2, 1, 0]))
    
    def test_conj_trans_complex_values(self):
        """Test conjugate transpose with complex values"""
        wann = init_wanndata(2, 3)
        wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        
        # Set values at R=(1,0,0)
        wann.ham[1, 0, 0] = 1.0 + 1.0j
        wann.ham[1, 0, 1] = 2.0 - 1.0j
        wann.ham[1, 1, 0] = 3.0 + 0.5j
        wann.ham[1, 1, 1] = 4.0 - 2.0j
        
        # Set corresponding values at R=(-1,0,0)
        wann.ham[2, 0, 0] = 5.0 + 0.5j
        wann.ham[2, 0, 1] = 6.0 - 1.5j
        wann.ham[2, 1, 0] = 7.0 + 2.0j
        wann.ham[2, 1, 1] = 8.0 - 1.0j
        
        wann_ct = get_conj_trans_of_ham(wann)
        
        # Check conjugate property
        assert np.allclose(wann_ct.ham[1, 0, 0], np.conj(wann.ham[2, 0, 0]))
        assert np.allclose(wann_ct.ham[1, 1, 0], np.conj(wann.ham[2, 0, 1]))


class TestTrsymmHam:
    """Test trsymm_ham function (time-reversal symmetry)"""
    
    def test_trsymm_no_soc_real(self):
        """Test time-reversal without SOC on real Hamiltonian"""
        wann = init_wanndata(2, 1)
        wann.rvec = [Vector(0, 0, 0)]
        wann.ham[0] = np.array([[1.0, 2.0], [2.0, 3.0]])
        
        # Create simple orbitals
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb_info = [
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0),
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0)
        ]
        wann_tr = trsymm_ham(wann, orb_info, flag_soc=0)
        
        # For real Hamiltonian without SOC, time-reversal is identity
        assert np.allclose(wann_tr.ham, wann.ham)
    
    def test_trsymm_no_soc_complex(self):
        """Test time-reversal without SOC on complex Hamiltonian"""
        wann = init_wanndata(2, 1)
        wann.rvec = [Vector(0, 0, 0)]
        wann.ham[0] = np.array([
            [1.0 + 1.0j, 2.0 - 0.5j],
            [2.0 + 0.5j, 3.0 - 1.0j]
        ])
        
        # Create simple orbitals
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb_info = [
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0),
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0)
        ]
        wann_tr = trsymm_ham(wann, orb_info, flag_soc=0)
        
        # Without SOC, time-reversal is just complex conjugation
        assert np.allclose(wann_tr.ham, np.conj(wann.ham))
    
    def test_trsymm_with_soc(self):
        """Test time-reversal with SOC"""
        # Create 2-orbital system with spin (4 total states)
        wann = init_wanndata(4, 1)
        wann.rvec = [Vector(0, 0, 0)]
        
        # Set up orbital info with spins
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb_info = []
        for i in range(2):  # 2 orbitals
            for ms in range(2):  # spin up/down
                orb = WannOrb(site=site, axis=axis, r=i, l=0, mr=0, ms=ms)
                orb_info.append(orb)
        
        # Set non-trivial Hamiltonian (off-diagonal elements between spins)
        wann.ham[0, 0, 1] = 1.0 + 1.0j  # coupling between orb0_up and orb0_down
        wann.ham[0, 1, 0] = 1.0 - 1.0j  # Hermitian conjugate
        
        wann_tr = trsymm_ham(wann, orb_info, flag_soc=1)
        
        # Check that result is computed (specific values depend on spin structure)
        assert wann_tr.ham.shape == wann.ham.shape
        # For this specific case with SOC, time-reversal should produce non-zero result
        assert np.sum(np.abs(wann_tr.ham)) > 0


class TestSymmetrizeHamiltonians:
    """Test symmetrize_hamiltonians function"""
    
    def test_symmetrize_identity(self):
        """Test symmetrization with single identity operation"""
        wann = init_wanndata(2, 1)
        wann.rvec = [Vector(0, 0, 0)]
        wann.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        wann.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        
        ham_list = [wann]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=1)
        
        # With single identity, result should be same as input
        assert np.allclose(ham_symm.ham, wann.ham)
    
    def test_symmetrize_average(self):
        """Test symmetrization averages correctly"""
        # Create two Hamiltonians
        wann1 = init_wanndata(2, 1)
        wann1.rvec = [Vector(0, 0, 0)]
        wann1.ham[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        wann1.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        
        wann2 = init_wanndata(2, 1)
        wann2.rvec = [Vector(0, 0, 0)]
        wann2.ham[0] = np.array([[3.0, 0.0], [0.0, 3.0]])
        wann2.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        
        ham_list = [wann1, wann2]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=2)
        
        # Result should be average: (1+3)/2 = 2
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        assert np.allclose(ham_symm.ham[0], expected)
    
    def test_symmetrize_expand_rvec(self):
        """Test R-vector expansion during symmetrization"""
        wann1 = init_wanndata(2, 1)
        wann1.rvec = [Vector(0, 0, 0)]
        wann1.ham[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        wann1.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        
        wann2 = init_wanndata(2, 2)
        wann2.rvec = [Vector(0, 0, 0), Vector(1, 0, 0)]
        wann2.ham[0] = np.array([[2.0, 0.0], [0.0, 2.0]])
        wann2.ham[1] = np.array([[0.5, 0.0], [0.0, 0.5]])
        wann2.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        wann2.hamflag[1] = np.ones((2, 2), dtype=np.int32)
        
        ham_list = [wann1, wann2]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=2, expand_rvec=True)
        
        # Should have 2 R-vectors after expansion
        assert ham_symm.nrpt == 2


class TestApplyHermitianSymmetry:
    """Test apply_hermitian_symmetry function"""
    
    def test_hermitian_already_hermitian(self):
        """Test Hermitian symmetry on already Hermitian matrix"""
        wann = init_wanndata(2, 3)
        wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        
        # Set Hermitian Hamiltonian
        wann.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        wann.ham[1] = np.array([[0.0, 1.0+1.0j], [0.0, 0.0]])
        wann.ham[2] = np.array([[0.0, 0.0], [1.0-1.0j, 0.0]])
        
        wann_h = apply_hermitian_symmetry(wann)
        
        # Should be unchanged (or very close)
        # At R=0, diagonal should be real and matrix Hermitian
        assert np.allclose(wann_h.ham[0], wann.ham[0])
    
    def test_hermitian_makes_hermitian(self):
        """Test that Hermitian symmetry produces Hermitian result"""
        wann = init_wanndata(2, 3)
        wann.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        
        # Set non-Hermitian values
        wann.ham[0] = np.array([[1.0, 2.0+1.0j], [3.0-0.5j, 2.0]])
        wann.ham[1] = np.array([[0.5, 1.0+1.0j], [0.2, 0.3]])
        wann.ham[2] = np.array([[0.6, 0.1], [1.5-2.0j, 0.4]])
        
        wann_h = apply_hermitian_symmetry(wann)
        
        # At R=0, should be Hermitian
        H_R0 = wann_h.ham[0]
        assert np.allclose(H_R0, H_R0.conj().T)


class TestCheckHamiltonianConsistency:
    """Test check_hamiltonian_consistency function"""
    
    def test_consistency_identical(self):
        """Test consistency of identical Hamiltonians"""
        wann1 = init_wanndata(2, 1)
        wann1.rvec = [Vector(0, 0, 0)]
        wann1.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        wann2 = init_wanndata(2, 1)
        wann2.rvec = [Vector(0, 0, 0)]
        wann2.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        consistent, max_diff = check_hamiltonian_consistency(wann1, wann2)
        
        assert consistent
        assert max_diff < 1e-10
    
    def test_consistency_within_tolerance(self):
        """Test consistency within tolerance"""
        wann1 = init_wanndata(2, 1)
        wann1.rvec = [Vector(0, 0, 0)]
        wann1.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        wann2 = init_wanndata(2, 1)
        wann2.rvec = [Vector(0, 0, 0)]
        wann2.ham[0] = np.array([[1.01, 0.51], [0.51, 2.01]])
        
        consistent, max_diff = check_hamiltonian_consistency(wann1, wann2, tolerance=0.1)
        
        assert consistent
        assert max_diff < 0.1
    
    def test_inconsistency_outside_tolerance(self):
        """Test inconsistency outside tolerance"""
        wann1 = init_wanndata(2, 1)
        wann1.rvec = [Vector(0, 0, 0)]
        wann1.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        wann2 = init_wanndata(2, 1)
        wann2.rvec = [Vector(0, 0, 0)]
        wann2.ham[0] = np.array([[2.0, 1.0], [1.0, 3.0]])
        
        consistent, max_diff = check_hamiltonian_consistency(wann1, wann2, tolerance=0.1)
        
        assert not consistent
        assert max_diff > 0.1
    
    def test_inconsistency_different_sizes(self):
        """Test inconsistency with different sizes"""
        wann1 = init_wanndata(2, 1)
        wann2 = init_wanndata(3, 1)
        
        consistent, max_diff = check_hamiltonian_consistency(wann1, wann2)
        
        assert not consistent
        assert max_diff == float('inf')


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_full_symmetrization_workflow(self):
        """Test complete symmetrization workflow"""
        # Create initial Hamiltonian
        wann = init_wanndata(2, 1)
        wann.rvec = [Vector(0, 0, 0)]
        wann.ham[0] = np.array([[1.0 + 0.1j, 0.5 - 0.2j], 
                                [0.6 + 0.1j, 2.0 - 0.15j]])
        wann.hamflag[0] = np.ones((2, 2), dtype=np.int32)
        
        # Apply Hermitian symmetry
        wann_h = apply_hermitian_symmetry(wann)
        
        # Check Hermiticity at R=0
        H_R0 = wann_h.ham[0]
        assert np.allclose(H_R0, H_R0.conj().T, atol=1e-10)
    
    def test_time_reversal_then_hermitian(self):
        """Test time-reversal followed by Hermitian symmetry"""
        wann = init_wanndata(2, 1)
        wann.rvec = [Vector(0, 0, 0)]
        wann.ham[0] = np.array([[1.0, 0.5j], [0.5j, 2.0]])
        
        # Create simple orbitals
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb_info = [
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0),
            WannOrb(site=site, axis=axis, r=0, l=0, mr=0, ms=0)
        ]
        
        # Apply time-reversal (without SOC)
        wann_tr = trsymm_ham(wann, orb_info, flag_soc=0)
        
        # Apply Hermitian symmetry
        wann_h = apply_hermitian_symmetry(wann_tr)
        
        # Result should be real and symmetric
        H = wann_h.ham[0]
        assert np.allclose(H.imag, 0, atol=1e-10)
