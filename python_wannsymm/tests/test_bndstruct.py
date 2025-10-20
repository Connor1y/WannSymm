"""
Tests for bndstruct module

Test band structure calculation including Fourier transform,
diagonalization, character analysis, and file output.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from wannsymm.bndstruct import (
    SmallGroup,
    diagonalize_hamiltonian,
    identify_degeneracies,
    write_band_characters,
    write_band_eigenvalues,
    write_bands
)
from wannsymm.wanndata import WannData
from wannsymm.vector import Vector
from wannsymm.constants import eps6


class TestSmallGroup:
    """Test SmallGroup data structure."""
    
    def test_initialization(self):
        """Test SmallGroup initialization."""
        sgrp = SmallGroup(order=0, element=[])
        assert sgrp.order == 0
        assert isinstance(sgrp.element, list)
    
    def test_with_elements(self):
        """Test SmallGroup with elements."""
        sgrp = SmallGroup(order=3, element=[0, 1, 2])
        assert sgrp.order == 3
        assert sgrp.element == [0, 1, 2]


class TestDiagonalizeHamiltonian:
    """Test Hamiltonian diagonalization and Fourier transform."""
    
    def test_simple_identity(self):
        """Test diagonalization of identity matrix at Gamma point."""
        # Create simple Hamiltonian: H = I (identity)
        hr = WannData(norb=2, nrpt=1)
        hr.ham[0] = np.eye(2, dtype=np.complex128)
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # Check eigenvalues (should be [1, 1])
        assert eig.shape == (2,)
        np.testing.assert_allclose(eig, [1.0, 1.0], rtol=1e-10)
        
        # Check eigenvectors (should be unitary)
        assert vec.shape == (2, 2)
        # Check orthonormality
        np.testing.assert_allclose(
            vec @ vec.T.conj(),
            np.eye(2),
            atol=1e-10
        )
    
    def test_diagonal_hamiltonian(self):
        """Test diagonalization of diagonal Hamiltonian."""
        # Create diagonal Hamiltonian
        hr = WannData(norb=3, nrpt=1)
        hr.ham[0] = np.diag([1.0, 2.0, 3.0]).astype(np.complex128)
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # Check eigenvalues (should be sorted [1, 2, 3])
        np.testing.assert_allclose(eig, [1.0, 2.0, 3.0], rtol=1e-10)
    
    def test_fourier_transform_two_sites(self):
        """Test Fourier transform with two R-points."""
        # Create Hamiltonian with hopping between two sites
        # H(R=0) = diag(1, 1), H(R=(1,0,0)) = off-diagonal hopping
        hr = WannData(norb=2, nrpt=2)
        
        # On-site energy at R=0
        hr.ham[0] = np.eye(2, dtype=np.complex128)
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        # Hopping at R=(1,0,0)
        hr.ham[1] = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
        hr.rvec[1] = Vector(1, 0, 0)
        hr.weight[1] = 1
        
        # At Gamma point (k=0), hopping adds constructively
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # H(k=0) = I + [[0, 0.5], [0.5, 0]] = [[1, 0.5], [0.5, 1]]
        # Eigenvalues should be 0.5 and 1.5
        np.testing.assert_allclose(sorted(eig), [0.5, 1.5], rtol=1e-10)
    
    def test_fourier_transform_phase(self):
        """Test that Fourier transform correctly includes phase factors."""
        # Single orbital, three R-points to ensure Hermiticity
        hr = WannData(norb=1, nrpt=3)
        
        hr.ham[0] = np.array([[0.0]], dtype=np.complex128)  # On-site at R=0
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        # Hopping at R=(1,0,0)
        hr.ham[1] = np.array([[1.0]], dtype=np.complex128)
        hr.rvec[1] = Vector(1, 0, 0)
        hr.weight[1] = 1
        
        # Hopping at R=(-1,0,0) to ensure Hermiticity
        hr.ham[2] = np.array([[1.0]], dtype=np.complex128)
        hr.rvec[2] = Vector(-1, 0, 0)
        hr.weight[2] = 1
        
        # At k=(0.5, 0, 0), phases cancel: exp(i*pi) + exp(-i*pi) = -1 + -1 = -2
        kpt = Vector(0.5, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        np.testing.assert_allclose(eig, [-2.0], atol=1e-10)
        
        # At k=(0, 0, 0), phases add: exp(0) + exp(0) = 1 + 1 = 2
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        np.testing.assert_allclose(eig, [2.0], atol=1e-10)
        
        # At k=(0.25, 0, 0): exp(i*pi/2) + exp(-i*pi/2) = i + (-i) = 0
        kpt = Vector(0.25, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        np.testing.assert_allclose(eig, [0.0], atol=1e-10)
    
    def test_hermiticity(self):
        """Test that diagonalized Hamiltonian is Hermitian."""
        # Create a non-diagonal but Hermitian Hamiltonian
        hr = WannData(norb=2, nrpt=1)
        hr.ham[0] = np.array([[1.0, 0.5j], [-0.5j, 2.0]], dtype=np.complex128)
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # All eigenvalues should be real
        assert np.allclose(eig.imag, 0)
        
        # Eigenvalues should be positive
        assert np.all(eig.real > 0)


class TestIdentifyDegeneracies:
    """Test degeneracy identification."""
    
    def test_no_degeneracy(self):
        """Test with non-degenerate eigenvalues."""
        eig = np.array([1.0, 2.0, 3.0, 4.0])
        ndegen = identify_degeneracies(eig)
        
        np.testing.assert_array_equal(ndegen, [1, 1, 1, 1])
    
    def test_full_degeneracy(self):
        """Test with all eigenvalues degenerate."""
        eig = np.array([1.0, 1.0, 1.0, 1.0])
        ndegen = identify_degeneracies(eig, degenerate_tolerance=eps6)
        
        np.testing.assert_array_equal(ndegen, [4, 4, 4, 4])
    
    def test_partial_degeneracy(self):
        """Test with partially degenerate eigenvalues."""
        eig = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
        ndegen = identify_degeneracies(eig)
        
        np.testing.assert_array_equal(ndegen, [3, 3, 3, 1, 1])
    
    def test_multiple_groups(self):
        """Test with multiple degenerate groups."""
        eig = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
        ndegen = identify_degeneracies(eig)
        
        np.testing.assert_array_equal(ndegen, [2, 2, 3, 3, 3, 1])
    
    def test_tolerance(self):
        """Test degeneracy detection with tolerance."""
        # Eigenvalues within tolerance
        eig = np.array([1.0, 1.0 + 1e-7, 2.0])
        ndegen = identify_degeneracies(eig, degenerate_tolerance=1e-6)
        
        # Should be identified as degenerate
        np.testing.assert_array_equal(ndegen, [2, 2, 1])
        
        # With stricter tolerance
        ndegen = identify_degeneracies(eig, degenerate_tolerance=1e-8)
        
        # Should not be degenerate
        np.testing.assert_array_equal(ndegen, [1, 1, 1])


class TestWriteBandCharacters:
    """Test writing band characters to file."""
    
    def test_write_basic(self):
        """Test basic file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fnout = os.path.join(tmpdir, "test_characters.dat")
            
            # Create test data
            sgrp = SmallGroup(order=2, element=[0, 1])
            kpt = Vector(0.0, 0.0, 0.0)
            lattice = np.eye(3)
            rotations = np.array([np.eye(3), np.eye(3)])
            translations = np.array([[0, 0, 0], [0, 0, 0]])
            TR = np.array([0, 0])
            eig_hk = np.array([1.0, 2.0])
            ndegen = np.array([1, 1])
            sym_chas = [
                np.array([1.0 + 0j, 1.0 + 0j]),
                np.array([1.0 + 0j, -1.0 + 0j])
            ]
            
            write_band_characters(
                fnout, 0, kpt, sgrp, lattice, rotations, translations,
                TR, eig_hk, ndegen, sym_chas, 2
            )
            
            # Check file was created
            assert os.path.exists(fnout)
            
            # Check content
            with open(fnout, 'r') as f:
                content = f.read()
                assert "kpt:" in content
                assert "SmallGroupOrder= 2" in content
    
    def test_write_degenerate(self):
        """Test writing with degenerate bands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fnout = os.path.join(tmpdir, "test_degenerate.dat")
            
            sgrp = SmallGroup(order=1, element=[0])
            kpt = Vector(0.5, 0.5, 0.5)
            lattice = np.eye(3)
            rotations = np.array([np.eye(3)])
            translations = np.array([[0, 0, 0]])
            TR = np.array([0])
            eig_hk = np.array([1.0, 1.0, 2.0])
            ndegen = np.array([2, 2, 1])
            sym_chas = [np.array([2.0 + 0j, 2.0 + 0j, 1.0 + 0j])]
            
            write_band_characters(
                fnout, 0, kpt, sgrp, lattice, rotations, translations,
                TR, eig_hk, ndegen, sym_chas, 3
            )
            
            assert os.path.exists(fnout)
            
            with open(fnout, 'r') as f:
                content = f.read()
                assert "ndegen =     2" in content


class TestWriteBandEigenvalues:
    """Test writing band eigenvalues to file."""
    
    def test_write_basic(self):
        """Test basic eigenvalue file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fnout = os.path.join(tmpdir, "test_eigenvalues.dat")
            
            sgrp = SmallGroup(order=2, element=[0, 1])
            kpt = Vector(0.0, 0.0, 0.0)
            lattice = np.eye(3)
            rotations = np.array([np.eye(3), np.eye(3)])
            translations = np.array([[0, 0, 0], [0, 0, 0]])
            TR = np.array([0, 0])
            eig_hk = np.array([1.0, 2.0])
            ndegen = np.array([1, 1])
            sym_eigs = [
                np.array([1.0 + 0j, 1.0 + 0j]),
                np.array([1.0 + 0j, -1.0 + 0j])
            ]
            
            write_band_eigenvalues(
                fnout, 0, kpt, sgrp, lattice, rotations, translations,
                TR, eig_hk, ndegen, sym_eigs, 2
            )
            
            assert os.path.exists(fnout)
            
            with open(fnout, 'r') as f:
                content = f.read()
                assert "kpt:" in content
                assert "SmallGroupOrder= 2" in content


class TestWriteBands:
    """Test writing band structure to file."""
    
    def test_write_single_path(self):
        """Test writing single k-path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fnout = os.path.join(tmpdir, "test_bands.dat")
            
            norb = 2
            nkpath = 1
            nk_per_kpath = 3
            lattice = np.eye(3)
            
            # Create k-points: Gamma -> X (0,0,0) -> (0.5,0,0)
            kvecs = [
                Vector(0.0, 0.0, 0.0),
                Vector(0.25, 0.0, 0.0),
                Vector(0.5, 0.0, 0.0)
            ]
            
            # Create band energies
            ebands = np.array([
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2]
            ])
            
            with open(fnout, 'w') as f:
                write_bands(f, norb, nkpath, nk_per_kpath, lattice, kvecs, ebands)
            
            assert os.path.exists(fnout)
            
            with open(fnout, 'r') as f:
                lines = f.readlines()
                # Should have header + 3 k-points per band + blank lines
                assert len(lines) > 0
                assert "k-path len" in lines[0]
    
    def test_write_multiple_paths(self):
        """Test writing multiple k-path segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fnout = os.path.join(tmpdir, "test_bands_multi.dat")
            
            norb = 2
            nkpath = 2
            nk_per_kpath = 2
            lattice = np.eye(3)
            
            # Two path segments
            kvecs = [
                Vector(0.0, 0.0, 0.0),
                Vector(0.5, 0.0, 0.0),
                Vector(0.5, 0.0, 0.0),
                Vector(0.5, 0.5, 0.0)
            ]
            
            ebands = np.array([
                [1.0, 2.0],
                [1.5, 2.5],
                [1.5, 2.5],
                [2.0, 3.0]
            ])
            
            with open(fnout, 'w') as f:
                write_bands(f, norb, nkpath, nk_per_kpath, lattice, kvecs, ebands)
            
            assert os.path.exists(fnout)


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_tight_binding_chain(self):
        """Test 1D tight-binding chain."""
        # Create 1D chain with nearest-neighbor hopping
        # H = sum_i [ -t * (|i><i+1| + |i+1><i|) ]
        # In Wannier basis with 2 orbitals: 0 and 1
        hr = WannData(norb=2, nrpt=3)
        
        # On-site at R=0 (no on-site energy)
        hr.ham[0] = np.zeros((2, 2), dtype=np.complex128)
        hr.rvec[0] = Vector(0, 0, 0)
        hr.weight[0] = 1
        
        # Hopping right: <0|H|R=1,1> = -1
        hr.ham[1] = np.array([[0, 0], [-1, 0]], dtype=np.complex128)
        hr.rvec[1] = Vector(1, 0, 0)
        hr.weight[1] = 1
        
        # Hopping left: <1|H|R=-1,0> = -1 (Hermitian conjugate)
        hr.ham[2] = np.array([[0, -1], [0, 0]], dtype=np.complex128)
        hr.rvec[2] = Vector(-1, 0, 0)
        hr.weight[2] = 1
        
        # Test at Gamma point
        kpt = Vector(0, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # H(k=0) = [[0, -1], [-1, 0]], eigenvalues = +/-1
        np.testing.assert_allclose(sorted(eig), [-1.0, 1.0], rtol=1e-10)
        
        # Test at zone boundary
        kpt = Vector(0.5, 0, 0)
        eig, vec = diagonalize_hamiltonian(hr, kpt)
        
        # At k=pi/a: exp(i*pi) = -1, exp(-i*pi) = -1
        # H(k=0.5) = [[0, 1], [1, 0]], eigenvalues still = +/-1
        np.testing.assert_allclose(sorted(eig), [-1.0, 1.0], rtol=1e-10)
    
    def test_full_workflow(self):
        """Test complete workflow: diagonalize -> identify degeneracies -> write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple Hamiltonian
            hr = WannData(norb=2, nrpt=1)
            hr.ham[0] = np.eye(2, dtype=np.complex128)
            hr.rvec[0] = Vector(0, 0, 0)
            hr.weight[0] = 1
            
            kpt = Vector(0, 0, 0)
            
            # Diagonalize
            eig, vec = diagonalize_hamiltonian(hr, kpt)
            
            # Identify degeneracies
            ndegen = identify_degeneracies(eig)
            
            # Write characters
            fnout = os.path.join(tmpdir, "full_test.dat")
            sgrp = SmallGroup(order=1, element=[0])
            lattice = np.eye(3)
            rotations = np.array([np.eye(3)])
            translations = np.array([[0, 0, 0]])
            TR = np.array([0])
            sym_chas = [np.ones(2, dtype=np.complex128)]
            
            write_band_characters(
                fnout, 0, kpt, sgrp, lattice, rotations, translations,
                TR, eig, ndegen, sym_chas, 2
            )
            
            assert os.path.exists(fnout)

