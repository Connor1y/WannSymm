"""
Tests for readinput module

Test input file parsing.
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np

from wannsymm.readinput import (
    parseline,
    str2boolean,
    parse_projections_block,
    parse_kpath_block,
    parse_kpts_block,
    readinput,
    InputData,
    InputParseError,
)
from wannsymm.vector import Vector


class TestParseline:
    """Tests for parseline function"""
    
    def test_basic_parsing(self):
        """Test basic tag=value parsing"""
        tag, arg = parseline("DFTcode = VASP")
        assert tag == "dftcode"
        assert arg == "VASP"
    
    def test_colon_separator(self):
        """Test tag:value parsing"""
        tag, arg = parseline("Spinors: T")
        assert tag == "spinors"
        assert arg == "T"
    
    def test_case_insensitive(self):
        """Test case-insensitive tag parsing"""
        tag, arg = parseline("SeedName = test", ignorecase=True)
        assert tag == "seedname"
        assert arg == "test"
    
    def test_case_sensitive(self):
        """Test case-sensitive tag parsing"""
        tag, arg = parseline("SeedName = test", ignorecase=False)
        assert tag == "SeedName"
        assert arg == "test"
    
    def test_quoted_values(self):
        """Test parsing quoted values"""
        tag, arg = parseline("SeedName='my_seed'")
        assert tag == "seedname"
        assert arg == "my_seed"
        
        tag, arg = parseline('SeedName="my_seed"')
        assert tag == "seedname"
        assert arg == "my_seed"
    
    def test_comment_removal(self):
        """Test comment removal"""
        tag, arg = parseline("DFTcode = VASP # This is a comment")
        assert tag == "dftcode"
        assert arg == "VASP"
        
        tag, arg = parseline("# Full line comment")
        assert tag == ""
        assert arg == ""
    
    def test_empty_line(self):
        """Test empty line handling"""
        tag, arg = parseline("")
        assert tag == ""
        assert arg == ""


class TestStr2Boolean:
    """Tests for str2boolean function"""
    
    def test_true_values(self):
        """Test parsing True values"""
        assert str2boolean("T") is True
        assert str2boolean("t") is True
        assert str2boolean("1") is True
        assert str2boolean(".True.") is True
        assert str2boolean(".TRUE.") is True
        assert str2boolean(".T.") is True
    
    def test_false_values(self):
        """Test parsing False values"""
        assert str2boolean("F") is False
        assert str2boolean("f") is False
        assert str2boolean("0") is False
        assert str2boolean(".False.") is False
        assert str2boolean(".FALSE.") is False
        assert str2boolean(".F.") is False
    
    def test_invalid_values(self):
        """Test invalid boolean values"""
        assert str2boolean("invalid") is None
        assert str2boolean("yes") is None
        assert str2boolean("") is None


class TestParseProjectionsBlock:
    """Tests for parse_projections_block function"""
    
    def test_simple_projections(self):
        """Test simple projection parsing"""
        lines = ["V: d", "O: p", "end projections"]
        projs, flag, n = parse_projections_block(lines)
        
        assert len(projs) == 2
        assert projs[0].element == "V"
        assert projs[0].orbitals == ["d"]
        assert projs[1].element == "O"
        assert projs[1].orbitals == ["p"]
        assert n == 3
    
    def test_multiple_orbitals(self):
        """Test projection with multiple orbitals"""
        lines = ["Mn: s;d", "end projections"]
        projs, flag, n = parse_projections_block(lines)
        
        assert len(projs) == 1
        assert projs[0].element == "Mn"
        assert projs[0].orbitals == ["s", "d"]
    
    def test_specific_orbitals(self):
        """Test specific orbital names"""
        lines = ["C: px,py,pz", "end projections"]
        projs, flag, n = parse_projections_block(lines)
        
        assert len(projs) == 1
        assert projs[0].orbitals == ["px", "py", "pz"]
    
    def test_missing_end_tag(self):
        """Test error on missing end tag"""
        lines = ["V: d", "O: p"]
        with pytest.raises(InputParseError):
            parse_projections_block(lines)


class TestParseKpathBlock:
    """Tests for parse_kpath_block function"""
    
    def test_simple_kpath(self):
        """Test simple k-path parsing"""
        lines = ["G 0 0 0 X 0.5 0 0", "end"]
        paths, labels, n = parse_kpath_block(lines)
        
        assert len(paths) == 1
        assert len(labels) == 1
        assert labels[0] == ("G", "X")
        assert paths[0][0].x == 0.0
        assert paths[0][1].x == 0.5
    
    def test_multiple_kpaths(self):
        """Test multiple k-paths"""
        lines = [
            "G 0 0 0 X 0.5 0 0",
            "X 0.5 0 0 M 0.5 0.5 0",
            "end"
        ]
        paths, labels, n = parse_kpath_block(lines)
        
        assert len(paths) == 2
        assert labels[0] == ("G", "X")
        assert labels[1] == ("X", "M")


class TestParseKptsBlock:
    """Tests for parse_kpts_block function"""
    
    def test_simple_kpts(self):
        """Test simple k-point parsing"""
        lines = ["0 0 0", "0.5 0 0", "end"]
        kpts, n = parse_kpts_block(lines)
        
        assert len(kpts) == 2
        assert kpts[0].x == 0.0
        assert kpts[1].x == 0.5


class TestReadinputMinimal:
    """Tests for readinput function with minimal input"""
    
    def test_minimal_input(self):
        """Test minimal valid input file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("Spinors = F\n")
            f.write("SeedName = 'test'\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.dftcode == "VASP"
            assert data.spinors is False
            assert data.seedname == "test"
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_default_values(self):
        """Test that default values are set"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.spinors is False  # Default
            assert data.global_trsymm is True  # Default
            assert data.nk_per_kpath == 100  # Default
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestReadinputFull:
    """Tests for readinput function with full input"""
    
    def test_all_boolean_tags(self):
        """Test all boolean tags"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("Spinors = T\n")
            f.write("SeedName = 'test'\n")
            f.write("restart = T\n")
            f.write("bands = F\n")
            f.write("chaeig = T\n")
            f.write("global_trsymm = F\n")
            f.write("expandrvec = T\n")
            f.write("everysymm = T\n")
            f.write("enforce_hermitian = T\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.spinors is True
            assert data.restart is True
            assert data.chaeig is True
            assert data.global_trsymm is False
            assert data.expandrvec is True
            assert data.everysymm is True
            assert data.hermitian is True
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_tolerances(self):
        """Test tolerance values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write("ham_tolerance = 0.05\n")
            f.write("degenerate_tolerance = 1e-5\n")
            f.write("symm_magnetic_tolerance = 1e-4\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.ham_tolerance == 0.05
            assert data.degenerate_tolerance == 1e-5
            assert data.symm_magnetic_tolerance == 1e-4
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_projections_parsing(self):
        """Test projections block parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write("begin projections\n")
            f.write("V: d\n")
            f.write("O: p\n")
            f.write("end projections\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert len(data.projections) == 2
            assert data.projections[0].element == "V"
            assert data.projections[0].orbitals == ["d"]
            assert data.projections[1].element == "O"
            assert data.projections[1].orbitals == ["p"]
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_kpoints_parsing(self):
        """Test k-point and k-path parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write("kpt = 0 0 0\n")
            f.write("begin kpath\n")
            f.write("G 0 0 0 X 0.5 0 0\n")
            f.write("end\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert len(data.kpts) == 1
            assert len(data.kpaths) == 1
            assert data.klabels[0] == ("G", "X")
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestPOSCARParsing:
    """Tests for POSCAR parsing"""
    
    def test_poscar_direct(self):
        """Test POSCAR parsing with Direct coordinates"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.poscar', delete=False) as f:
            f.write("Test structure\n")
            f.write("1.0\n")
            f.write("3.0 0.0 0.0\n")
            f.write("0.0 3.0 0.0\n")
            f.write("0.0 0.0 3.0\n")
            f.write("Sr V O\n")
            f.write("1 1 3\n")
            f.write("Direct\n")
            f.write("0.5 0.5 0.5\n")
            f.write("0.0 0.0 0.0\n")
            f.write("0.5 0.0 0.0\n")
            f.write("0.0 0.5 0.0\n")
            f.write("0.0 0.0 0.5\n")
            poscar_file = f.name
        
        # Create input file that references this POSCAR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write(f"Use_POSCAR = '{poscar_file}'\n")
            input_file = f.name
        
        try:
            data = readinput(input_file)
            assert data.lattice is not None
            assert data.lattice.shape == (3, 3)
            assert data.atom_positions is not None
            assert data.atom_positions.shape[0] == 5  # 1+1+3 atoms
            assert data.atom_types == ["Sr", "V", "O"]
            assert data.num_atoms_each == [1, 1, 3]
        finally:
            Path(poscar_file).unlink(missing_ok=True)
            Path(input_file).unlink(missing_ok=True)
    
    def test_poscar_cartesian_error(self):
        """Test that Cartesian coordinates work with ASE"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.poscar', delete=False) as f:
            f.write("Test structure\n")
            f.write("1.0\n")
            f.write("3.0 0.0 0.0\n")
            f.write("0.0 3.0 0.0\n")
            f.write("0.0 0.0 3.0\n")
            f.write("Sr\n")
            f.write("1\n")
            f.write("Cartesian\n")
            f.write("1.5 1.5 1.5\n")
            poscar_file = f.name
        
        # Create input file that references this POSCAR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write(f"Use_POSCAR = '{poscar_file}'\n")
            input_file = f.name
        
        try:
            # ASE should handle Cartesian coordinates
            data = readinput(input_file)
            assert data.lattice is not None
            assert data.atom_positions is not None
        finally:
            Path(poscar_file).unlink(missing_ok=True)
            Path(input_file).unlink(missing_ok=True)


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_missing_file(self):
        """Test error on missing input file"""
        with pytest.raises(FileNotFoundError):
            readinput("nonexistent.in")
    
    def test_invalid_boolean(self):
        """Test error on invalid boolean value"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = VASP\n")
            f.write("SeedName = 'test'\n")
            f.write("Spinors = invalid\n")
            temp_file = f.name
        
        try:
            with pytest.raises(InputParseError):
                readinput(temp_file)
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_invalid_dftcode(self):
        """Test error on invalid DFT code"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("DFTcode = UNKNOWN\n")
            f.write("SeedName = 'test'\n")
            temp_file = f.name
        
        try:
            with pytest.raises(InputParseError):
                readinput(temp_file)
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestComments:
    """Tests for comment handling"""
    
    def test_hash_comments(self):
        """Test # comments"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("DFTcode = VASP  # inline comment\n")
            f.write("SeedName = 'test'\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.dftcode == "VASP"
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_exclamation_comments(self):
        """Test ! comments"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("! This is a comment\n")
            f.write("DFTcode = VASP  ! inline comment\n")
            f.write("SeedName = 'test'\n")
            temp_file = f.name
        
        try:
            data = readinput(temp_file)
            assert data.dftcode == "VASP"
        finally:
            Path(temp_file).unlink(missing_ok=True)
