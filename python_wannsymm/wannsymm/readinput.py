"""
Input file reading module for WannSymm

Parses wannsymm.in input file and POSCAR structure files.
Translated from: src/readinput.h and src/readinput.c

Translation Status: ⏳ NOT STARTED
"""

# TODO: Translate from src/readinput.h and src/readinput.c
#
# Original C files: src/readinput.h (157 lines) + src/readinput.c (1,673 lines)
# Total lines: 1,830 (LARGEST MODULE)
# Complexity: High
# Estimated effort: 20-30 hours
#
# Translation notes:
# - This is the most complex module - break into smaller functions
# - Use configparser or custom parser for input file
# - Consider using ASE (Atomic Simulation Environment) for POSCAR reading
# - Support all input tags documented in README.md
# - Maintain backward compatibility with C version input format
#
# Key functionality:
# 1. Input file parsing (wannsymm.in format)
#    - DFTcode tag (VASP/QE)
#    - Spinors tag (T/F)
#    - SeedName tag
#    - Use_POSCAR tag
#    - projections block
#    - MAGMOM tag
#    - kpath block
#    - Many optional tags (see README.md)
#
# 2. POSCAR reading
#    - Parse VASP POSCAR format
#    - Extract lattice vectors
#    - Extract atomic positions
#    - Extract atomic species
#
# 3. Projections parsing
#    - Parse Wannier90 projection format
#    - Examples: "Mn: s;d", "F: p"
#    - Handle different orbital types (s, p, d, f)
#
# 4. K-path parsing
#    - Parse begin/end kpath blocks
#    - Store k-point paths for band structure
#
# 5. Structure validation
#    - Check required tags exist
#    - Validate tag values
#    - Set default values for optional tags
#
# Functions to translate (many):
# - read_input: Main input reading function
# - parse_tag_*: Individual tag parsers
# - read_poscar: POSCAR file reader
# - parse_projections: Projection block parser
# - parse_kpath: K-path block parser
# - validate_input: Input validation
# - Many helper functions
#
# Note: This module will need extensive testing due to complexity
