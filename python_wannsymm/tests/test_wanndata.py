"""
Tests for wanndata module

Test Hamiltonian data structures and I/O.
"""

import pytest
import os
import tempfile
import numpy as np

from wannsymm.wanndata import (
    WannData,
    init_wanndata,
    read_ham,
    write_ham,
    write_reduced_ham,
    find_index_of_ham,
    combine_wanndata,
)
from wannsymm.vector import Vector
from wannsymm.wannorb import WannOrb


class TestWannDataInit:
    """Test WannData initialization."""
    
    def test_init_wanndata_basic(self):
        """Test basic initialization."""
        wann = init_wanndata(norb=4, nrpt=10)
        
        assert wann.norb == 4
        assert wann.nrpt == 10
        assert wann.ham.shape == (10, 4, 4)
        assert wann.hamflag.shape == (10, 4, 4)
        assert len(wann.rvec) == 10
        assert wann.weight.shape == (10,)
    
    def test_init_wanndata_arrays_initialized(self):
        """Test that arrays are properly initialized."""
        wann = init_wanndata(norb=2, nrpt=3)
        
        # Check ham is zero
        assert np.allclose(wann.ham, 0.0)
        
        # Check hamflag is zero
        assert np.all(wann.hamflag == 0)
        
        # Check weights are 1
        assert np.all(wann.weight == 1)
    
    def test_wanndata_dataclass_init(self):
        """Test direct dataclass initialization."""
        wann = WannData(norb=2, nrpt=1)
        
        assert wann.norb == 2
        assert wann.nrpt == 1
        assert wann.ham.shape == (1, 2, 2)


class TestWannDataFileIO:
    """Test file I/O operations."""
    
    def test_write_read_roundtrip(self):
        """Test round-trip write and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = os.path.join(tmpdir, "test")
            
            # Create test data
            wann_orig = init_wanndata(norb=2, nrpt=2)
            wann_orig.rvec[0] = Vector(0, 0, 0)
            wann_orig.rvec[1] = Vector(1, 0, 0)
            wann_orig.ham[0, 0, 0] = 1.0 + 0.5j
            wann_orig.ham[0, 0, 1] = 0.2 + 0.3j
            wann_orig.ham[1, 1, 0] = 0.5 - 0.1j
            wann_orig.weight[0] = 1
            wann_orig.weight[1] = 2
            
            # Write to file
            write_ham(wann_orig, seed)
            
            # Read back
            wann_read = read_ham(seed)
            
            # Compare
            assert wann_read.norb == wann_orig.norb
            assert wann_read.nrpt == wann_orig.nrpt
            assert np.allclose(wann_read.ham, wann_orig.ham)
            assert np.array_equal(wann_read.weight, wann_orig.weight)
            
            for i in range(wann_orig.nrpt):
                assert wann_read.rvec[i].x == wann_orig.rvec[i].x
                assert wann_read.rvec[i].y == wann_orig.rvec[i].y
                assert wann_read.rvec[i].z == wann_orig.rvec[i].z
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_ham("nonexistent_file")
    
    def test_write_ham_format(self):
        """Test that written file has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = os.path.join(tmpdir, "test")
            
            wann = init_wanndata(norb=2, nrpt=1)
            wann.rvec[0] = Vector(0, 0, 0)
            wann.ham[0, 0, 0] = 1.5 + 0.5j
            wann.weight[0] = 1
            
            write_ham(wann, seed)
            
            # Check file content
            with open(f"{seed}_hr.dat", 'r') as f:
                lines = f.readlines()
            
            # Line 1: header
            assert "symmetrized" in lines[0].lower() or "hamiltonian" in lines[0].lower()
            
            # Line 2: norb
            assert int(lines[1].strip()) == 2
            
            # Line 3: nrpt
            assert int(lines[2].strip()) == 1
    
    def test_write_reduced_ham(self):
        """Test writing reduced Hamiltonian."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = os.path.join(tmpdir, "test")
            
            wann = init_wanndata(norb=3, nrpt=1)
            wann.rvec[0] = Vector(0, 0, 0)
            # Set only diagonal elements (should be 3 nonzero)
            wann.ham[0, 0, 0] = 1.0
            wann.ham[0, 1, 1] = 2.0
            wann.ham[0, 2, 2] = 3.0
            
            write_reduced_ham(wann, seed)
            
            # Check file exists
            assert os.path.exists(f"{seed}_hr.dat")


class TestWannDataLookup:
    """Test element lookup functions."""
    
    def test_find_index_of_ham_valid(self):
        """Test finding valid Hamiltonian element."""
        wann = init_wanndata(norb=2, nrpt=1)
        
        # Create orbital info
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb2 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=1)
        orb_info = [orb1, orb2]
        
        # Find index
        idx = find_index_of_ham(
            wann, orb_info, 0,
            site, 1, 0, 0, 0,  # First orbital
            site, 1, 0, 0, 1   # Second orbital
        )
        
        # Should return valid index
        assert idx >= 0
        # For norb=2, irpt=0, jorb=1, iorb=0: index = 0*4 + 1*2 + 0 = 2
        assert idx == 2
    
    def test_find_index_of_ham_invalid_rvec(self):
        """Test with invalid R-vector index."""
        wann = init_wanndata(norb=2, nrpt=1)
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb_info = [orb1]
        
        # Invalid R-vector index
        idx = find_index_of_ham(
            wann, orb_info, -1,
            site, 1, 0, 0, 0,
            site, 1, 0, 0, 0
        )
        
        assert idx == -1
    
    def test_find_index_of_ham_orbital_not_found(self):
        """Test when orbital is not found."""
        wann = init_wanndata(norb=2, nrpt=1)
        site = Vector(0, 0, 0)
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        orb1 = WannOrb(site=site, axis=axis, r=1, l=0, mr=0, ms=0)
        orb_info = [orb1]
        
        # Search for non-existent orbital (different quantum numbers)
        idx = find_index_of_ham(
            wann, orb_info, 0,
            site, 1, 0, 0, 0,
            site, 2, 0, 0, 0  # Different r quantum number
        )
        
        assert idx == -3  # Ket orbital not found


class TestCombineWannData:
    """Test combining spin-up and spin-down Hamiltonians."""
    
    def test_combine_wanndata_basic(self):
        """Test basic combination of spin channels."""
        # Create spin-up Hamiltonian
        wup = init_wanndata(norb=2, nrpt=2)
        wup.rvec[0] = Vector(0, 0, 0)
        wup.rvec[1] = Vector(1, 0, 0)
        wup.ham[0, 0, 0] = 1.0
        wup.ham[1, 1, 1] = 2.0
        wup.weight[0] = 1
        wup.weight[1] = 2
        
        # Create spin-down Hamiltonian
        wdn = init_wanndata(norb=2, nrpt=2)
        wdn.rvec[0] = Vector(0, 0, 0)
        wdn.rvec[1] = Vector(1, 0, 0)
        wdn.ham[0, 0, 0] = 3.0
        wdn.ham[1, 1, 1] = 4.0
        wdn.weight[0] = 1
        wdn.weight[1] = 2
        
        # Combine
        wcomb = combine_wanndata(wup, wdn)
        
        # Check dimensions
        assert wcomb.norb == 4
        assert wcomb.nrpt == 2
        
        # Check spin-up block
        assert wcomb.ham[0, 0, 0] == 1.0
        assert wcomb.ham[1, 1, 1] == 2.0
        
        # Check spin-down block
        assert wcomb.ham[0, 2, 2] == 3.0
        assert wcomb.ham[1, 3, 3] == 4.0
        
        # Check off-diagonal blocks are zero
        assert wcomb.ham[0, 0, 2] == 0.0
        assert wcomb.ham[0, 2, 0] == 0.0
    
    def test_combine_wanndata_mismatched_nrpt(self):
        """Test that mismatched nrpt raises error."""
        wup = init_wanndata(norb=2, nrpt=2)
        wdn = init_wanndata(norb=2, nrpt=3)
        
        with pytest.raises(ValueError, match="same nrpt"):
            combine_wanndata(wup, wdn)
    
    def test_combine_wanndata_mismatched_rvec(self):
        """Test that mismatched R-vectors raise error."""
        wup = init_wanndata(norb=2, nrpt=2)
        wdn = init_wanndata(norb=2, nrpt=2)
        
        wup.rvec[0] = Vector(0, 0, 0)
        wup.rvec[1] = Vector(1, 0, 0)
        
        wdn.rvec[0] = Vector(0, 0, 0)
        wdn.rvec[1] = Vector(0, 1, 0)  # Different!
        
        with pytest.raises(ValueError, match="R-vectors don't match"):
            combine_wanndata(wup, wdn)


class TestWannDataEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_wanndata(self):
        """Test creation with zero size."""
        wann = WannData()
        assert wann.norb == 0
        assert wann.nrpt == 0
    
    def test_large_norb(self):
        """Test with large number of orbitals."""
        wann = init_wanndata(norb=100, nrpt=5)
        assert wann.ham.shape == (5, 100, 100)
    
    def test_complex_hamiltonian_elements(self):
        """Test storing complex Hamiltonian elements."""
        wann = init_wanndata(norb=2, nrpt=1)
        
        # Set various complex values
        wann.ham[0, 0, 0] = 1.5 + 2.3j
        wann.ham[0, 0, 1] = -0.5 + 0.8j
        wann.ham[0, 1, 0] = -0.5 - 0.8j  # Hermitian conjugate
        wann.ham[0, 1, 1] = 2.0
        
        assert np.real(wann.ham[0, 0, 0]) == 1.5
        assert np.imag(wann.ham[0, 0, 0]) == 2.3
        assert np.real(wann.ham[0, 1, 1]) == 2.0
        assert np.imag(wann.ham[0, 1, 1]) == 0.0
    
    def test_hamflag_functionality(self):
        """Test hamflag enable/disable functionality."""
        wann = init_wanndata(norb=2, nrpt=1)
        
        # Initially all disabled
        assert np.all(wann.hamflag == 0)
        
        # Enable specific elements
        wann.hamflag[0, 0, 0] = 1
        wann.hamflag[0, 1, 1] = 1
        
        assert wann.hamflag[0, 0, 0] == 1
        assert wann.hamflag[0, 0, 1] == 0


class TestWannDataConsistency:
    """Test consistency and validation."""
    
    def test_weight_array_length(self):
        """Test weight array has correct length."""
        wann = init_wanndata(norb=3, nrpt=5)
        assert len(wann.weight) == 5
    
    def test_rvec_list_length(self):
        """Test R-vector list has correct length."""
        wann = init_wanndata(norb=3, nrpt=7)
        assert len(wann.rvec) == 7
    
    def test_roundtrip_preserves_data(self):
        """Test that write/read preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = os.path.join(tmpdir, "test")
            
            # Create complex test data
            wann1 = init_wanndata(norb=3, nrpt=3)
            
            # Set R-vectors
            wann1.rvec[0] = Vector(0, 0, 0)
            wann1.rvec[1] = Vector(1, 0, 0)
            wann1.rvec[2] = Vector(-1, 1, 0)
            
            # Set weights
            wann1.weight[0] = 1
            wann1.weight[1] = 2
            wann1.weight[2] = 4
            
            # Set Hamiltonian elements
            for irpt in range(3):
                for i in range(3):
                    for j in range(3):
                        wann1.ham[irpt, i, j] = (irpt + 1) * (i + 1) * (j + 1) + 0.5j * irpt
            
            # Write and read
            write_ham(wann1, seed)
            wann2 = read_ham(seed)
            
            # Compare all data
            assert wann2.norb == wann1.norb
            assert wann2.nrpt == wann1.nrpt
            assert np.allclose(wann2.ham, wann1.ham, rtol=1e-14)
            assert np.array_equal(wann2.weight, wann1.weight)
            
            for i in range(wann1.nrpt):
                assert wann2.rvec[i].x == wann1.rvec[i].x
                assert wann2.rvec[i].y == wann1.rvec[i].y
                assert wann2.rvec[i].z == wann1.rvec[i].z
