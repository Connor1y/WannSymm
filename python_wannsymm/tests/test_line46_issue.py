"""
Test for line 46 issue in wannier90.up_symmed_hr.dat

The issue is that H(R=(-3,-3,-5), i=7, j=1) should be zero but is non-zero.
This represents Mn2(orbital 7=s) to Mn1(orbital 1=s) hopping at that R-vector.
"""

import pytest
import os
import shutil
from pathlib import Path
import numpy as np

from wannsymm.readinput import readinput
from wannsymm.readsymm import readsymm
from wannsymm.wanndata import read_ham, write_ham
from wannsymm.main import run_symmetrization


class TestLine46Issue:
    """Test for the specific line 46 issue in MnF2 example."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.repo_root = Path(__file__).parent.parent.parent
        cls.example_dir = cls.repo_root / "examples" / "mnf2"
        cls.input_dir = cls.example_dir / "input"
        cls.output_dir = cls.example_dir / "output"
        
        if not cls.input_dir.exists():
            pytest.skip("mnf2 input directory not found")
        if not cls.output_dir.exists():
            pytest.skip("mnf2 reference output directory not found")
    
    def test_mnf2_line46_value(self, tmp_path):
        """Test that line 46 (H(R=(-3,-3,-5), 7, 1)) is zero as in reference."""
        # Copy input files
        for file in ['POSCAR', 'wannier90.up.in', 'wannier90.up_hr.dat']:
            shutil.copy(self.input_dir / file, tmp_path)
        
        # Copy symmetries file
        shutil.copy(self.output_dir / 'symmetries.dat', tmp_path)
        
        # Change to tmp_path
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Run symmetrization
            input_data = readinput('wannier90.up.in')
            symm_data = readsymm('symmetries.dat')
            ham_in, ham_symm = run_symmetrization(input_data, symm_data, Path('wannsymm.out'))
            write_ham(ham_symm, 'wannier90.up_symmed')
            
            # Read the output file and check line 46
            with open('wannier90.up_symmed_hr.dat', 'r') as f:
                for i, line in enumerate(f, 1):
                    if i == 46:
                        parts = line.split()
                        assert len(parts) >= 7
                        rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
                        orb_i, orb_j = int(parts[3]), int(parts[4])
                        real_part, imag_part = float(parts[5]), float(parts[6])
                        
                        # Check this is the right element
                        assert (rx, ry, rz) == (-3, -3, -5), f"Line 46 should be R=(-3,-3,-5), got ({rx},{ry},{rz})"
                        assert orb_i == 7 and orb_j == 1, f"Line 46 should be orbitals (7,1), got ({orb_i},{orb_j})"
                        
                        # Check the value is zero (with small tolerance)
                        tolerance = 1e-6
                        assert abs(real_part) < tolerance, (
                            f"Line 46 real part should be ~0.0, got {real_part}. "
                            f"This is H(R=(-3,-3,-5), Mn2-s, Mn1-s) which should be forbidden by symmetry."
                        )
                        assert abs(imag_part) < tolerance, (
                            f"Line 46 imag part should be ~0.0, got {imag_part}"
                        )
                        break
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    # Run the test
    pytest.main([__file__, "-v", "-s"])
