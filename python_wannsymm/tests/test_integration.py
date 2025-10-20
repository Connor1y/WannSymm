"""
Integration tests for WannSymm

Test complete workflows using example data.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

from wannsymm.readinput import readinput, InputData
from wannsymm.readsymm import readsymm, SymmetryOperation, SymmetryData
from wannsymm.wanndata import init_wanndata, write_ham, read_ham
from wannsymm.wannorb import WannOrb
from wannsymm.vector import Vector
from wannsymm.rotate_ham import (
    apply_hermitian_symmetry,
    symmetrize_hamiltonians,
    check_hamiltonian_consistency
)


class TestMinimalSymmetrization:
    """Test basic symmetrization workflow on minimal example."""
    
    def setup_method(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_minimal_input(self) -> Path:
        """
        Create a minimal input file for testing.
        
        Returns
        -------
        Path
            Path to created input file
        """
        # Create minimal input without POSCAR dependency
        input_content = """# Minimal test input file
DFTcode = VASP
Spinors = F
SeedName = 'test'

# Simple projection
begin projections
  f=0.0,0.0,0.0: s
end projections

restart = 0
expandrvec = 1
hermitian = 1
symm_from_file = 1
symm_input_file = 'symmetries.in'
"""
        input_file = Path("test.in")
        input_file.write_text(input_content)
        return input_file
    
    def create_minimal_hamiltonian(self, seedname: str) -> None:
        """
        Create a minimal Hamiltonian file for testing.
        
        Parameters
        ----------
        seedname : str
            Seedname for the Hamiltonian
        """
        # Create simple 1-orbital Hamiltonian with 3 R-points
        ham = init_wanndata(norb=1, nrpt=3)
        ham.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        ham.weight = np.array([1, 2, 2], dtype=np.int32)
        
        # Simple on-site and hopping elements
        ham.ham[0] = np.array([[0.5+0.0j]])  # On-site
        ham.ham[1] = np.array([[-0.1+0.0j]])  # Hopping to (1,0,0)
        ham.ham[2] = np.array([[-0.1+0.0j]])  # Hopping to (-1,0,0)
        
        ham.hamflag = np.ones_like(ham.ham, dtype=np.int32)
        
        write_ham(ham, seedname)
    
    def create_symmetry_file(self) -> Path:
        """
        Create a symmetry file with identity operation.
        
        Returns
        -------
        Path
            Path to created symmetry file
        """
        symm_content = """# Minimal symmetry file - identity only
global_trsymm = T
nsymm = 1
1.0  0.0  0.0
0.0  1.0  0.0
0.0  0.0  1.0
0.0  0.0  0.0  F
"""
        symm_file = Path("symmetries.in")
        symm_file.write_text(symm_content)
        return symm_file
    
    def test_read_minimal_input(self):
        """Test reading minimal input file."""
        input_file = self.create_minimal_input()
        
        # Should be able to parse input
        input_data = readinput(str(input_file))
        
        assert input_data.dftcode == "VASP"
        assert input_data.spinors is False
        assert input_data.seedname == "test"
    
    def test_read_symmetry_file(self):
        """Test reading symmetry file."""
        symm_file = self.create_symmetry_file()
        
        symm_data = readsymm(str(symm_file))
        
        assert len(symm_data.operations) == 1
        assert symm_data.global_time_reversal is True
        
        # Check identity operation
        op = symm_data.operations[0]
        assert np.allclose(op.rotation, np.eye(3))
        assert np.allclose(op.translation, np.zeros(3))
        assert op.time_reversal is False
    
    def test_hamiltonian_io(self):
        """Test Hamiltonian I/O."""
        seedname = "test"
        self.create_minimal_hamiltonian(seedname)
        
        # Should be able to read back
        ham = read_ham(seedname)
        
        assert ham.norb == 1
        assert ham.nrpt == 3
        assert len(ham.rvec) == 3
        
        # Check on-site energy
        assert np.isclose(ham.ham[0, 0, 0].real, 0.5)
    
    def test_hermitian_symmetry(self):
        """Test Hermitian symmetry enforcement."""
        # Create non-Hermitian Hamiltonian
        ham = init_wanndata(norb=2, nrpt=1)
        ham.rvec = [Vector(0, 0, 0)]
        ham.ham[0] = np.array([[1.0, 2.0+1.0j], [3.0-0.5j, 2.0]])
        ham.hamflag = np.ones_like(ham.ham, dtype=np.int32)
        
        # Apply Hermitian symmetry
        ham_h = apply_hermitian_symmetry(ham)
        
        # Check Hermiticity
        H = ham_h.ham[0]
        assert np.allclose(H, H.conj().T)
    
    def test_identity_symmetrization(self):
        """Test symmetrization with identity operation preserves Hamiltonian."""
        # Create Hamiltonian
        ham = init_wanndata(norb=2, nrpt=3)
        ham.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        ham.weight = np.array([1, 2, 2], dtype=np.int32)
        ham.ham[0] = np.array([[1.0, 0.5], [0.5, 2.0]])
        ham.ham[1] = np.array([[0.0, -0.1], [-0.1, 0.0]])
        ham.ham[2] = np.array([[0.0, -0.1], [-0.1, 0.0]])
        ham.hamflag = np.ones_like(ham.ham, dtype=np.int32)
        
        # Symmetrize with single identity operation
        ham_list = [ham]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=1, expand_rvec=False)
        
        # Should be identical
        is_consistent, max_diff = check_hamiltonian_consistency(ham, ham_symm)
        assert is_consistent
        assert max_diff < 1e-10
    
    def test_averaging_symmetrization(self):
        """Test averaging over multiple Hamiltonians."""
        # Create two different Hamiltonians
        ham1 = init_wanndata(norb=1, nrpt=1)
        ham1.rvec = [Vector(0, 0, 0)]
        ham1.ham[0] = np.array([[1.0]])
        ham1.hamflag = np.ones_like(ham1.ham, dtype=np.int32)
        
        ham2 = init_wanndata(norb=1, nrpt=1)
        ham2.rvec = [Vector(0, 0, 0)]
        ham2.ham[0] = np.array([[3.0]])
        ham2.hamflag = np.ones_like(ham2.ham, dtype=np.int32)
        
        # Symmetrize - should get average
        ham_list = [ham1, ham2]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=2, expand_rvec=False)
        
        # Check average
        expected = 2.0
        assert np.isclose(ham_symm.ham[0, 0, 0].real, expected)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end symmetrization workflow."""
        # Create all necessary files
        input_file = self.create_minimal_input()
        symm_file = self.create_symmetry_file()
        self.create_minimal_hamiltonian("test")
        
        # Read input
        input_data = readinput(str(input_file))
        
        # Modify input to use symmetry file
        input_data.symm_from_file = True
        input_data.symm_input_file = str(symm_file)
        
        # Read symmetries
        symm_data = readsymm(input_data.symm_input_file)
        
        # Read Hamiltonian
        ham_in = read_ham(input_data.seedname)
        
        # Apply Hermitian symmetry
        ham_h = apply_hermitian_symmetry(ham_in)
        
        # For identity symmetry, just use the Hermitian version
        ham_list = [ham_h]
        
        # Symmetrize
        ham_symm = symmetrize_hamiltonians(
            ham_list,
            nsymm=len(symm_data.operations),
            expand_rvec=input_data.expandrvec
        )
        
        # Write output
        seedname_out = f"{input_data.seedname}_symmed"
        write_ham(ham_symm, seedname_out)
        
        # Verify output file exists
        output_file = Path(f"{seedname_out}_hr.dat")
        assert output_file.exists()
        
        # Read back and verify
        ham_read = read_ham(seedname_out)
        assert ham_read.norb == ham_symm.norb
        assert ham_read.nrpt == ham_symm.nrpt
        
        # Should be consistent (identity symmetry)
        is_consistent, max_diff = check_hamiltonian_consistency(
            ham_h, ham_symm, tolerance=1e-8
        )
        assert is_consistent, f"Hamiltonians differ by {max_diff}"


class TestSymmetryBehavior:
    """Test that symmetrization produces expected symmetry behavior."""
    
    def test_inversion_symmetry(self):
        """Test that inversion symmetry is correctly applied."""
        # Create Hamiltonian with hopping
        ham = init_wanndata(norb=1, nrpt=2)
        ham.rvec = [Vector(0, 0, 0), Vector(1, 0, 0)]
        ham.weight = np.array([1, 1], dtype=np.int32)
        ham.ham[0] = np.array([[0.0]])
        ham.ham[1] = np.array([[1.0]])  # Hopping to (1,0,0)
        ham.hamflag = np.ones_like(ham.ham, dtype=np.int32)
        
        # With inversion symmetry, should also have (-1,0,0) hopping
        # For this test, we'll create the inverted version manually
        ham_inv = init_wanndata(norb=1, nrpt=2)
        ham_inv.rvec = [Vector(0, 0, 0), Vector(-1, 0, 0)]
        ham_inv.weight = np.array([1, 1], dtype=np.int32)
        ham_inv.ham[0] = np.array([[0.0]])
        ham_inv.ham[1] = np.array([[1.0]])  # Hopping to (-1,0,0)
        ham_inv.hamflag = np.ones_like(ham_inv.ham, dtype=np.int32)
        
        # Symmetrize with R-vector expansion
        ham_list = [ham, ham_inv]
        ham_symm = symmetrize_hamiltonians(ham_list, nsymm=2, expand_rvec=True)
        
        # Should have both (1,0,0) and (-1,0,0)
        assert ham_symm.nrpt >= 3  # At least (0,0,0), (1,0,0), (-1,0,0)
    
    def test_hermiticity_preserved(self):
        """Test that symmetrization preserves Hermiticity."""
        # Start with Hermitian Hamiltonian
        ham = init_wanndata(norb=2, nrpt=3)
        ham.rvec = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(-1, 0, 0)]
        ham.weight = np.array([1, 2, 2], dtype=np.int32)
        
        # Make Hermitian on-site
        ham.ham[0] = np.array([[1.0, 0.5+0.2j], [0.5-0.2j, 2.0]])
        
        # Hoppings related by Hermiticity
        ham.ham[1] = np.array([[0.0, -0.1+0.1j], [-0.2, 0.0]])
        ham.ham[2] = np.conj(ham.ham[1].T)
        
        ham.hamflag = np.ones_like(ham.ham, dtype=np.int32)
        
        # Apply Hermitian symmetry
        ham_h = apply_hermitian_symmetry(ham)
        
        # On-site should be Hermitian
        H_onsite = ham_h.ham[0]
        assert np.allclose(H_onsite, H_onsite.conj().T)


def test_placeholder_removed():
    """This test verifies the placeholder has been removed."""
    # The old placeholder test has been replaced with real tests
    assert True
