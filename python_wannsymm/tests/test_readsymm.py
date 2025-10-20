"""
Tests for readsymm module

Test symmetry operations reading and validation.
"""

import pytest
import numpy as np
import tempfile
import os

from wannsymm.readsymm import (
    SymmetryOperation,
    SymmetryData,
    validate_rotation_matrix,
    readsymm,
    SymmetryError,
)
from wannsymm.readinput import InputParseError


class TestSymmetryOperation:
    """Test SymmetryOperation dataclass."""
    
    def test_create_identity_operation(self):
        """Test creating identity symmetry operation."""
        rot = np.eye(3)
        trans = np.zeros(3)
        op = SymmetryOperation(rot, trans, False)
        
        assert op.rotation.shape == (3, 3)
        assert op.translation.shape == (3,)
        assert not op.time_reversal
        assert np.allclose(op.rotation, np.eye(3))
    
    def test_invalid_rotation_shape(self):
        """Test that invalid rotation shape raises error."""
        with pytest.raises(ValueError, match="must be 3x3"):
            SymmetryOperation(np.eye(2), np.zeros(3), False)
    
    def test_invalid_translation_shape(self):
        """Test that invalid translation shape raises error."""
        with pytest.raises(ValueError, match="must have 3 elements"):
            SymmetryOperation(np.eye(3), np.zeros(2), False)


class TestValidateRotationMatrix:
    """Test rotation matrix validation."""
    
    def test_identity_is_valid(self):
        """Identity matrix is a valid rotation."""
        validate_rotation_matrix(np.eye(3))
    
    def test_rotation_90_z_is_valid(self):
        """90-degree rotation around z-axis is valid."""
        rot = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=float)
        validate_rotation_matrix(rot)
    
    def test_inversion_is_valid(self):
        """Inversion (-I) is valid (det = -1)."""
        validate_rotation_matrix(-np.eye(3))
    
    def test_mirror_plane_is_valid(self):
        """Mirror plane is valid (det = -1)."""
        rot = np.array([
            [1,  0, 0],
            [0,  1, 0],
            [0,  0, -1]
        ], dtype=float)
        validate_rotation_matrix(rot)
    
    def test_non_orthogonal_raises_error(self):
        """Non-orthogonal matrix should raise error."""
        rot = np.array([
            [1, 0, 0],
            [0, 2, 0],  # Not orthogonal
            [0, 0, 1]
        ], dtype=float)
        
        with pytest.raises(SymmetryError, match="not orthogonal"):
            validate_rotation_matrix(rot)
    
    def test_wrong_determinant_raises_error(self):
        """Matrix with det != ±1 should raise error."""
        rot = np.array([
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5]
        ], dtype=float)
        
        # This matrix is also not orthogonal, so that error will be caught first
        with pytest.raises(SymmetryError, match="not orthogonal"):
            validate_rotation_matrix(rot)
    
    def test_wrong_shape_raises_error(self):
        """Wrong shape should raise error."""
        with pytest.raises(SymmetryError, match="must be 3x3"):
            validate_rotation_matrix(np.eye(2))


class TestReadsymm:
    """Test readsymm function with various file formats."""
    
    def test_simple_symmetry_file(self):
        """Test parsing a simple symmetry file with one operation."""
        content = """
# Test symmetry file
global_trsymm = T
nsymm = 1
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 1
            assert symm_data.global_time_reversal
            assert np.allclose(symm_data.operations[0].rotation, np.eye(3))
            assert np.allclose(symm_data.operations[0].translation, np.zeros(3))
            assert not symm_data.operations[0].time_reversal
        finally:
            os.unlink(fname)
    
    def test_multiple_symmetries(self):
        """Test parsing multiple symmetry operations."""
        content = """
nsymm = 2
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0
-1  0  0
 0 -1  0
 0  0 -1
0.5 0.5 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 2
            assert np.allclose(symm_data.operations[0].rotation, np.eye(3))
            assert np.allclose(symm_data.operations[1].rotation, -np.eye(3))
            assert np.allclose(symm_data.operations[1].translation, [0.5, 0.5, 0.5])
        finally:
            os.unlink(fname)
    
    def test_time_reversal_in_operation(self):
        """Test time-reversal flag in individual operation."""
        content = """
global_trsymm = T
nsymm = 1
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0 T
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 1
            assert symm_data.operations[0].time_reversal
            # Global TR should be disabled when any operation has TR
            assert not symm_data.global_time_reversal
        finally:
            os.unlink(fname)
    
    def test_no_time_reversal_with_f_flag(self):
        """Test that F flag correctly disables time-reversal."""
        content = """
nsymm = 1
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0 F
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert not symm_data.operations[0].time_reversal
        finally:
            os.unlink(fname)
    
    def test_global_trsymm_variants(self):
        """Test different tag names for global time-reversal symmetry."""
        for tag in ['global_trsymm', 'trsymm', 'globaltime-reversalsymmetry']:
            content = f"""
{tag} = F
nsymm = 1
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0
"""
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
                f.write(content)
                fname = f.name
            
            try:
                symm_data = readsymm(fname)
                assert not symm_data.global_time_reversal, f"Failed for tag: {tag}"
            finally:
                os.unlink(fname)
    
    def test_missing_nsymm_raises_error(self):
        """Test that missing nsymm tag raises error."""
        content = """
global_trsymm = T
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            with pytest.raises(InputParseError, match="Missing 'nsymm'"):
                readsymm(fname)
        finally:
            os.unlink(fname)
    
    def test_truncated_file_raises_error(self):
        """Test that truncated file raises error."""
        content = """
nsymm = 2
1 0 0
0 1 0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            with pytest.raises(InputParseError, match="Unexpected end of file"):
                readsymm(fname)
        finally:
            os.unlink(fname)
    
    def test_invalid_rotation_matrix(self):
        """Test that invalid rotation matrix raises error."""
        content = """
nsymm = 1
2 0 0
0 2 0
0 0 2
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            with pytest.raises(SymmetryError, match="Invalid rotation matrix"):
                readsymm(fname)
        finally:
            os.unlink(fname)
    
    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled."""
        content = """
nsymm = 1
2 0 0
0 2 0
0 0 2
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            # Should not raise when validation is disabled
            symm_data = readsymm(fname, validate=False)
            assert len(symm_data.operations) == 1
        finally:
            os.unlink(fname)
    
    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(InputParseError, match="Cannot open"):
            readsymm("/nonexistent/file.symm")


class TestCubicSymmetries:
    """Test symmetries for known crystal systems."""
    
    def test_cubic_identity(self):
        """Test identity operation in cubic system."""
        content = """
# Cubic symmetry - identity only
nsymm = 1
1 0 0
0 1 0
0 0 1
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 1
            rot = symm_data.operations[0].rotation
            
            # Verify orthogonality
            assert np.allclose(rot.T @ rot, np.eye(3))
            # Verify determinant
            assert np.isclose(np.linalg.det(rot), 1.0)
        finally:
            os.unlink(fname)
    
    def test_cubic_rotations(self):
        """Test common cubic rotations (90° around axes)."""
        # Rotation 90° around z-axis
        rot_z_90 = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=float)
        
        # Rotation 90° around x-axis
        rot_x_90 = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=float)
        
        # Rotation 90° around y-axis
        rot_y_90 = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ], dtype=float)
        
        for name, rot in [("z", rot_z_90), ("x", rot_x_90), ("y", rot_y_90)]:
            content = f"""
# Rotation 90° around {name}-axis
nsymm = 1
{rot[0,0]} {rot[0,1]} {rot[0,2]}
{rot[1,0]} {rot[1,1]} {rot[1,2]}
{rot[2,0]} {rot[2,1]} {rot[2,2]}
0.0 0.0 0.0
"""
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
                f.write(content)
                fname = f.name
            
            try:
                symm_data = readsymm(fname)
                assert len(symm_data.operations) == 1
                
                parsed_rot = symm_data.operations[0].rotation
                assert np.allclose(parsed_rot, rot), f"Failed for {name}-axis rotation"
                
                # Verify properties
                assert np.allclose(parsed_rot.T @ parsed_rot, np.eye(3))
                assert np.isclose(np.linalg.det(parsed_rot), 1.0)
            finally:
                os.unlink(fname)


class TestSpglibIntegration:
    """Test spglib integration for symmetry finding."""
    
    def test_spglib_available(self):
        """Test that spglib is available."""
        try:
            import spglib
            assert spglib is not None
        except ImportError:
            pytest.skip("spglib not available")
    
    def test_fcc_lattice_symmetries(self):
        """Test that FCC lattice has correct number of symmetries using spglib."""
        try:
            import spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # FCC lattice (conventional cubic cell)
        lattice = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ]) * 4.0  # Scale to 4 Angstrom lattice parameter
        
        # Single atom at origin
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = [1]  # Atomic numbers
        
        cell = (lattice, positions, numbers)
        
        # Get symmetry dataset from spglib
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        
        assert dataset is not None
        # FCC has 48 symmetries in the primitive cell
        assert len(dataset['rotations']) >= 12  # At least 12 for basic point group
    
    def test_simple_cubic_lattice(self):
        """Test simple cubic lattice symmetries with spglib."""
        try:
            import spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = [1]
        
        cell = (lattice, positions, numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        
        assert dataset is not None
        # Simple cubic should have Oh point group (48 symmetries)
        assert len(dataset['rotations']) == 48
        
        # Verify that all rotations are valid
        for rot in dataset['rotations']:
            rot_float = rot.astype(np.float64)
            validate_rotation_matrix(rot_float)
    
    def test_hexagonal_lattice(self):
        """Test hexagonal lattice symmetries with spglib."""
        try:
            import spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # Hexagonal lattice
        a = 3.0
        c = 5.0
        lattice = np.array([
            [a, 0, 0],
            [-a/2, a*np.sqrt(3)/2, 0],
            [0, 0, c]
        ])
        
        positions = np.array([[1/3, 2/3, 0.5]])
        numbers = [1]
        
        cell = (lattice, positions, numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        
        assert dataset is not None
        # Hexagonal should have 6-fold or higher symmetry
        assert len(dataset['rotations']) >= 6
        
        # Note: spglib returns rotation matrices in fractional coordinates
        # To validate in Cartesian space, need to transform: R_cart = L @ R_frac @ L^-1
        # For simplicity, just verify that rotations are integer matrices with det = ±1
        for rot in dataset['rotations']:
            det = np.linalg.det(rot)
            assert np.isclose(abs(det), 1.0), f"Rotation determinant should be ±1, got {det}"


class TestIntegrationWithKnownLattices:
    """Test against known crystallographic data."""
    
    def test_identity_plus_inversion(self):
        """Test system with identity and inversion (centrosymmetric)."""
        content = """
# Centrosymmetric system
global_trsymm = T
nsymm = 2
1  0  0
0  1  0
0  0  1
0.0 0.0 0.0
-1  0  0
 0 -1  0
 0  0 -1
0.0 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 2
            assert symm_data.global_time_reversal
            
            # First should be identity
            assert np.allclose(symm_data.operations[0].rotation, np.eye(3))
            # Second should be inversion
            assert np.allclose(symm_data.operations[1].rotation, -np.eye(3))
        finally:
            os.unlink(fname)
    
    def test_screw_axis_with_translation(self):
        """Test screw axis symmetry with fractional translation."""
        # 2-fold screw along z-axis: rotate 180° + translate (0, 0, 0.5)
        rot_180_z = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ], dtype=float)
        
        content = """
nsymm = 1
-1  0  0
 0 -1  0
 0  0  1
0.0 0.0 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 1
            assert np.allclose(symm_data.operations[0].rotation, rot_180_z)
            assert np.allclose(symm_data.operations[0].translation, [0.0, 0.0, 0.5])
        finally:
            os.unlink(fname)
    
    def test_glide_plane(self):
        """Test glide plane symmetry (mirror + translation)."""
        # Mirror in xy-plane + translation along x
        content = """
nsymm = 1
1  0  0
0  1  0
0  0 -1
0.5 0.0 0.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.symm') as f:
            f.write(content)
            fname = f.name
        
        try:
            symm_data = readsymm(fname)
            
            assert len(symm_data.operations) == 1
            expected_rot = np.diag([1, 1, -1])
            assert np.allclose(symm_data.operations[0].rotation, expected_rot)
            assert np.allclose(symm_data.operations[0].translation, [0.5, 0.0, 0.0])
        finally:
            os.unlink(fname)


class TestFindSymmetriesWithSpglib:
    """Test find_symmetries_with_spglib function."""
    
    def test_simple_cubic_structure(self):
        """Test symmetry detection for simple cubic structure."""
        try:
            from wannsymm.readsymm import find_symmetries_with_spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # Simple cubic lattice
        lattice = np.eye(3) * 4.0
        atom_positions = np.array([[0.0, 0.0, 0.0]])
        atom_types = ['Fe']
        num_atoms_each = [1]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                output_file=output_file
            )
            
            # Simple cubic should have 48 symmetries (Oh point group)
            assert len(symm_data.operations) == 48
            assert symm_data.global_time_reversal is True
            
            # Verify output file was created and contains expected content
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                content = f.read()
                assert 'nsymm = 48' in content
                assert 'space group infomation:' in content
                assert 'global time-reversal symmetry = True' in content
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_fcc_structure(self):
        """Test symmetry detection for FCC structure."""
        try:
            from wannsymm.readsymm import find_symmetries_with_spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # FCC lattice (primitive cell)
        lattice = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ]) * 4.0
        
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
                output_file=output_file
            )
            
            # FCC primitive cell should have 48 symmetries
            assert len(symm_data.operations) == 48
            
            # Verify rotation matrices are integer with det = ±1
            # Note: spglib returns rotations in fractional coordinates
            # These are not orthogonal in Cartesian space, but preserve the lattice
            for op in symm_data.operations:
                det = np.linalg.det(op.rotation)
                assert np.isclose(abs(det), 1.0), f"Rotation determinant should be ±1, got {det}"
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_output_file_format(self):
        """Test that output file has correct format matching C version."""
        try:
            from wannsymm.readsymm import find_symmetries_with_spglib
        except ImportError:
            pytest.skip("spglib not available")
        
        # Simple structure for testing
        lattice = np.eye(3) * 5.0
        atom_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        atom_types = ['Na', 'Cl']
        num_atoms_each = [1, 1]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            output_file = f.name
        
        try:
            symm_data = find_symmetries_with_spglib(
                lattice=lattice,
                atom_positions=atom_positions,
                atom_types=atom_types,
                num_atoms_each=num_atoms_each,
                output_file=output_file
            )
            
            # Read the output file back and verify it can be parsed
            symm_data_read = readsymm(output_file)
            
            # Should have same number of operations
            assert len(symm_data_read.operations) == len(symm_data.operations)
            
            # Verify rotations and translations match
            for i in range(len(symm_data.operations)):
                assert np.allclose(
                    symm_data_read.operations[i].rotation,
                    symm_data.operations[i].rotation
                )
                assert np.allclose(
                    symm_data_read.operations[i].translation,
                    symm_data.operations[i].translation
                )
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
