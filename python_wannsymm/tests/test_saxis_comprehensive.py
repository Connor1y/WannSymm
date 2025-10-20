"""
Comprehensive test demonstrating the SAXIS transformation fix.

This test shows that after the fix, the Python implementation correctly
transforms magnetic moments for arbitrary SAXIS orientations.
"""

import pytest
import numpy as np
from wannsymm.readinput import apply_saxis_transformation


def rotation_matrix_zaxis_to_direction(direction):
    """
    Compute rotation matrix that rotates z-axis to given direction.
    
    This is the reference implementation using standard rotation matrices.
    """
    direction = np.array(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    # Spherical angles
    r = np.linalg.norm(direction)
    theta = np.arccos(direction[2] / r)  # polar angle from z
    phi = np.arctan2(direction[1], direction[0])  # azimuthal angle
    
    # Rotation matrix: Rz(phi) * Ry(theta)
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1]
    ])
    
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    return Rz @ Ry


class TestSaxisTransformationMath:
    """Verify SAXIS transformation matches mathematical expectation."""
    
    def test_saxis_along_cardinal_directions(self):
        """Test SAXIS along x, y, z axes."""
        test_cases = [
            ((0.0, 0.0, 1.0), [0.0, 0.0, 4.6]),  # z-axis (identity)
            ((1.0, 0.0, 0.0), [4.6, 0.0, 0.0]),  # x-axis
            ((0.0, 1.0, 0.0), [0.0, 4.6, 0.0]),  # y-axis
        ]
        
        for saxis, expected in test_cases:
            magmom_in = [4.6]  # z-component in spin frame
            result = apply_saxis_transformation(magmom_in, 1, 0, saxis)
            
            np.testing.assert_allclose(
                result[0], expected, atol=1e-10,
                err_msg=f"Failed for SAXIS={saxis}"
            )
    
    def test_saxis_arbitrary_directions(self):
        """Test SAXIS in arbitrary directions."""
        test_directions = [
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
        ]
        
        for direction in test_directions:
            # Normalize
            saxis = np.array(direction, dtype=np.float64)
            saxis = tuple(saxis / np.linalg.norm(saxis))
            
            # Magnetic moment in spin frame (along z)
            magmom_spin = 4.6
            
            # Expected: rotate z-vector to saxis direction
            expected = np.array(saxis) * magmom_spin
            
            # Apply transformation
            result = apply_saxis_transformation([magmom_spin], 1, 0, saxis)
            
            np.testing.assert_allclose(
                result[0], expected, atol=1e-10,
                err_msg=f"Failed for SAXIS={saxis}"
            )
    
    def test_rotation_matrix_consistency(self):
        """Verify our transformation matches standard rotation matrix."""
        directions = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        
        for direction in directions:
            # Get rotation matrix
            R = rotation_matrix_zaxis_to_direction(direction)
            
            # Vector in spin frame
            v_spin = np.array([0.0, 0.0, 4.6])
            
            # Expected result from rotation matrix
            expected = R @ v_spin
            
            # Our transformation
            saxis = tuple(direction)
            result = apply_saxis_transformation([4.6], 1, 0, saxis)
            
            np.testing.assert_allclose(
                result[0], expected, atol=1e-10,
                err_msg=f"Failed for direction={direction}"
            )
    
    def test_multiple_atoms(self):
        """Test with multiple atoms."""
        saxis = (1.0, 1.0, 0.0)
        magmom_in = [4.6, -4.6, 0.0, 2.3]
        
        result = apply_saxis_transformation(magmom_in, 4, 0, saxis)
        
        # Each should be rotated to saxis direction (or opposite)
        direction = np.array(saxis) / np.linalg.norm(saxis)
        
        expected = np.array([
            4.6 * direction,
            -4.6 * direction,
            0.0 * direction,
            2.3 * direction,
        ])
        
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_soc_mode_passthrough(self):
        """Test SOC mode with 3D input passes through correctly for default SAXIS."""
        magmom_in = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        saxis = (0.0, 0.0, 1.0)
        
        result = apply_saxis_transformation(magmom_in, 2, 1, saxis)
        
        # For default SAXIS with SOC, should be identity
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_bug_demonstration(self):
        """
        Demonstrate the bug was real by showing what the old formula would give.
        
        This is a documentation test showing why the fix was necessary.
        """
        # For SAXIS = (1, 0, 0), we have alpha=0, beta=pi/2
        # Input: maxis = (0, 0, 4.6)
        
        alpha = 0.0
        beta = np.pi / 2.0
        maxis = np.array([0.0, 0.0, 4.6])
        
        # BUGGY formula (what C code had):
        buggy_y = (np.cos(beta)*np.sin(alpha)*maxis[0] + 
                  np.cos(alpha)*maxis[1] + 
                  np.sin(beta)*np.cos(alpha)*maxis[2])
        
        # CORRECT formula (what we fixed):
        correct_y = (np.cos(beta)*np.sin(alpha)*maxis[0] + 
                    np.cos(alpha)*maxis[1] + 
                    np.sin(beta)*np.sin(alpha)*maxis[2])
        
        # Buggy gave 4.6, correct gives 0
        assert abs(buggy_y - 4.6) < 1e-10, "Buggy formula should give 4.6"
        assert abs(correct_y - 0.0) < 1e-10, "Correct formula should give 0"
        
        # Verify our implementation uses the correct formula
        result = apply_saxis_transformation([4.6], 1, 0, (1.0, 0.0, 0.0))
        assert abs(result[0, 1] - 0.0) < 1e-10, "Our implementation should use correct formula"


if __name__ == "__main__":
    # Run demonstration
    test = TestSaxisTransformationMath()
    
    print("Testing SAXIS transformation fix...")
    print("\n1. Testing cardinal directions:")
    test.test_saxis_along_cardinal_directions()
    print("   ✓ x, y, z axes work correctly")
    
    print("\n2. Testing arbitrary directions:")
    test.test_saxis_arbitrary_directions()
    print("   ✓ All arbitrary directions work correctly")
    
    print("\n3. Testing rotation matrix consistency:")
    test.test_rotation_matrix_consistency()
    print("   ✓ Matches standard rotation matrix formula")
    
    print("\n4. Testing multiple atoms:")
    test.test_multiple_atoms()
    print("   ✓ Multiple atoms handled correctly")
    
    print("\n5. Testing SOC mode:")
    test.test_soc_mode_passthrough()
    print("   ✓ SOC mode works correctly")
    
    print("\n6. Demonstrating the bug:")
    test.test_bug_demonstration()
    print("   ✓ Confirmed: old formula was wrong, new formula is correct")
    
    print("\n✅ All tests passed! SAXIS transformation is correctly implemented.")
