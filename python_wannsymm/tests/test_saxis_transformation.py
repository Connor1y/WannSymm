"""
Tests for SAXIS transformation

Verify that the Python implementation matches the C code behavior
for magnetic moment transformations.
"""

import pytest
import numpy as np

from wannsymm.readinput import apply_saxis_transformation


class TestSaxisTransformation:
    """Test SAXIS transformation replicates C code behavior."""
    
    def test_default_saxis_non_soc(self):
        """Test default SAXIS (0,0,1) with non-SOC."""
        magmom_array = [4.6, -4.6, 0.0, 0.0]
        natom = 4
        flag_soc = 0
        saxis = (0.0, 0.0, 1.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # For default SAXIS with non-SOC, values should go to z-component
        assert result.shape == (4, 3)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 4.6], atol=1e-10)
        np.testing.assert_allclose(result[1], [0.0, 0.0, -4.6], atol=1e-10)
        np.testing.assert_allclose(result[2], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result[3], [0.0, 0.0, 0.0], atol=1e-10)
    
    def test_x_axis_saxis_non_soc(self):
        """Test SAXIS along x-axis (1,0,0) with non-SOC."""
        magmom_array = [4.6]
        natom = 1
        flag_soc = 0
        saxis = (1.0, 0.0, 0.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # For SAXIS=(1,0,0), alpha=0, beta=pi/2
        # z-component should rotate to x-component
        # Corrected formula: x=4.6, y=0, z=0
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 0], 4.6, atol=1e-10)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-10)
    
    def test_y_axis_saxis_non_soc(self):
        """Test SAXIS along y-axis (0,1,0) with non-SOC."""
        magmom_array = [4.6]
        natom = 1
        flag_soc = 0
        saxis = (0.0, 1.0, 0.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # For SAXIS=(0,1,0), alpha=pi/2, beta=pi/2
        # Corrected formula: x=0, y=4.6, z=0
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 1], 4.6, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-10)
    
    def test_soc_mode(self):
        """Test with SOC mode (3 values per atom)."""
        # Each atom has 3 components
        magmom_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        natom = 2
        flag_soc = 1
        saxis = (0.0, 0.0, 1.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # For default SAXIS with SOC, should be identity transformation
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [4.0, 5.0, 6.0], atol=1e-10)
    
    def test_angles_calculation(self):
        """Test that angles are calculated correctly."""
        # Test SAXIS = (1, 1, 0)
        # alpha should be atan(1/1) = pi/4
        # beta should be atan(sqrt(2)/0) = pi/2
        magmom_array = [4.6]
        natom = 1
        flag_soc = 0
        saxis = (1.0, 1.0, 0.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # With alpha=pi/4, beta=pi/2:
        # x = cos(pi/2)*cos(pi/4)*0 - sin(pi/4)*0 + sin(pi/2)*cos(pi/4)*4.6 = 4.6/sqrt(2)
        # y = cos(pi/2)*sin(pi/4)*0 + cos(pi/4)*0 + sin(pi/2)*sin(pi/4)*4.6 = 4.6/sqrt(2) (corrected)
        # z = -sin(pi/2)*0 + cos(pi/2)*4.6 = 0
        assert result.shape == (1, 3)
        expected_val = 4.6 / np.sqrt(2)
        np.testing.assert_allclose(result[0, 0], expected_val, atol=1e-10)
        np.testing.assert_allclose(result[0, 1], expected_val, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-10)
    
    def test_zero_magmom(self):
        """Test with zero magnetic moments."""
        magmom_array = [0.0, 0.0]
        natom = 2
        flag_soc = 0
        saxis = (1.0, 0.0, 0.0)
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, np.zeros((2, 3)), atol=1e-10)
    
    def test_special_case_x_near_zero(self):
        """Test special case when SAXIS.x is near zero."""
        magmom_array = [4.6]
        natom = 1
        flag_soc = 0
        saxis = (1e-11, 1.0, 0.0)  # x < 1e-10, y != 0
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # alpha should be pi/2 (special case)
        # beta should be pi/2
        assert result.shape == (1, 3)
        # This should give similar result to y-axis case
    
    def test_special_case_z_near_zero(self):
        """Test special case when SAXIS.z is near zero."""
        magmom_array = [4.6]
        natom = 1
        flag_soc = 0
        saxis = (1.0, 0.0, 1e-11)  # z < 1e-10
        
        result = apply_saxis_transformation(magmom_array, natom, flag_soc, saxis)
        
        # beta should be pi/2 (special case)
        assert result.shape == (1, 3)
