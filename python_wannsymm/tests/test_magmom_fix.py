"""
Tests for magmom dimension handling fix

Test that magmom can be provided as 1D (one value per atom) or 2D (Nx3) arrays.
"""

import pytest
import numpy as np
import tempfile
import os

try:
    from wannsymm.readsymm import find_symmetries_with_spglib
    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False


@pytest.mark.skipif(not SPGLIB_AVAILABLE, reason="spglib not available")
class TestMagmomDimensions:
    """Test handling of magmom in different dimensions."""
    
    def test_magmom_1d_simple_cubic(self):
        """Test that 1D magmom (one value per atom) works correctly."""
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Fe']
        num_atoms_each = [1]
        
        # 1D magmom - one value per atom (represents z-component)
        magmom = [2.0]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            # This should not raise an error anymore
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=magmom,
                output_file=output_file
            )
            
            # Should have found some symmetries
            assert len(symm_data.operations) > 0
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_magmom_1d_multiple_atoms(self):
        """Test 1D magmom with multiple atoms."""
        # Simple cubic with 2 atoms
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        atom_types = ['Fe', 'Fe']
        num_atoms_each = [2]
        
        # 1D magmom - different values for each atom
        magmom = [2.0, -2.0]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=magmom,
                output_file=output_file
            )
            
            # Antiferromagnetic ordering should reduce symmetries
            assert len(symm_data.operations) > 0
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_magmom_2d_format(self):
        """Test that 2D magmom (Nx3) also works correctly."""
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Fe']
        num_atoms_each = [1]
        
        # 2D magmom - Nx3 format with x, y, z components
        magmom = np.array([[0.0, 0.0, 2.0]])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=magmom,
                output_file=output_file
            )
            
            # Should have found some symmetries
            assert len(symm_data.operations) > 0
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_magmom_zero_values(self):
        """Test that zero magmom values don't cause issues."""
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Cu']
        num_atoms_each = [1]
        
        # 1D magmom with zero value (non-magnetic)
        magmom = [0.0]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=magmom,
                output_file=output_file
            )
            
            # Non-magnetic should have full symmetry (48 for simple cubic)
            assert len(symm_data.operations) == 48
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_magmom_none(self):
        """Test that None magmom works correctly (non-magnetic case)."""
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Cu']
        num_atoms_each = [1]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=None,
                output_file=output_file
            )
            
            # Non-magnetic should have full symmetry (48 for simple cubic)
            assert len(symm_data.operations) == 48
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_magmom_as_list(self):
        """Test that magmom as Python list works correctly."""
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Fe']
        num_atoms_each = [1]
        
        # Python list (as it comes from readinput)
        magmom = [2.0]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                magmom=magmom,
                output_file=output_file
            )
            
            assert len(symm_data.operations) > 0
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
