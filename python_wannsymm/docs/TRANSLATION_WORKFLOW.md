# WannSymm Python Translation Workflow

This document provides step-by-step instructions and prompts for translating each C module to Python.

## Overview

The translation follows a bottom-up approach, starting with foundational modules and progressing to higher-level functionality. Each module should be:

1. Translated one-to-one from C to Python
2. Tested with comprehensive unit tests
3. Validated against the C version where possible
4. Documented with docstrings

## Translation Phases

### Phase 1: Foundation (Basic Data Structures)

#### Module 1: constants.py

**Priority**: HIGHEST - Foundation for all modules  
**Complexity**: Trivial  
**Estimated Time**: 0.5 hours  
**Dependencies**: None  
**C Files**: `src/constants.h`

**Translation Prompt**:
```
Translate src/constants.h to Python module wannsymm/constants.py:

1. Define all constants as module-level variables
2. Use UPPER_CASE naming for constants
3. Replace C types with Python equivalents:
   - double → float
   - int → int  
   - dcomplex → np.complex128
   - cmplx_i → 1j

4. Constants to translate:
   - MAXLEN = 512
   - MEDIUMLEN = 256
   - SHORTLEN = 128
   - MAXN = 10
   - eps4 = 1e-4
   - eps5 = 1e-5
   - eps6 = 1e-6
   - eps7 = 1e-7
   - eps8 = 1e-8
   - sqrt2 = 1.41421356237309504880168872421
   - PI = 3.14159265358979323846264338328
   - MAX_NUM_of_SYMM = 1536
   - MAX_NUM_of_atoms = 1024
   - MAX_L = 3

5. Add docstrings explaining each constant's purpose
6. Add type hints where appropriate
```

**Test Prompt**:
```
Create tests/test_constants.py:

1. Test all constants are defined
2. Verify types (float, int, complex)
3. Check mathematical constants have correct values
4. Verify tolerance values are positive and decreasing
5. Check array size limits are reasonable (>0)
```

---

#### Module 2: vector.py

**Priority**: HIGH - Used by almost all modules  
**Complexity**: Low-Medium  
**Estimated Time**: 4-6 hours  
**Dependencies**: constants.py  
**C Files**: `src/vector.h` (66 lines) + `src/vector.c` (346 lines)

**Translation Prompt**:
```
Translate src/vector.h and src/vector.c to wannsymm/vector.py:

1. Create Vector class using dataclass:
   @dataclass
   class Vector:
       x: float
       y: float
       z: float

2. Create Symm class for symmetry operations:
   @dataclass
   class Symm:
       rot: np.ndarray  # 3x3 rotation matrix
       trans: Vector    # translation vector

3. Implement VecLList class for linked list (or use Python list)

4. Translate all functions from vector.c:
   - init_vector() → Vector constructor
   - equal(), equale() → __eq__ method and equals() function
   - distance() - with optional lattice parameter
   - cross_product(), dot_product(), volume_product()
   - vector_scale(), vector_multiply(), vector_add(), vector_sub()
   - translate_match(), isenclosed(), find_vector()
   - vector_rotate(), vector_Rotate()
   - vector_norm(), vector_normalization(), vector_round()
   - array2vector(), kpt_equivalent(), vec_comp()
   - vec_llist_* functions

5. Use numpy for efficient operations where possible
6. Add type hints to all functions
7. Add docstrings in NumPy style
8. Keep function names similar to C version for traceability

Implementation notes:
- Use numpy arrays internally for performance
- Implement operator overloading (__add__, __sub__, __mul__)
- Make Vector immutable or handle mutability carefully
- Consider using numpy.allclose for floating-point comparisons
```

**Test Prompt**:
```
Create tests/test_vector.py:

1. Test Vector creation and properties
2. Test vector arithmetic (add, sub, scale, multiply)
3. Test vector products (dot, cross, volume)
4. Test vector operations (norm, normalize, rotate)
5. Test vector comparison with different tolerances
6. Test linked list operations (add, delete, find, pop)
7. Test edge cases:
   - Zero vectors
   - Collinear vectors
   - Very small/large magnitudes
8. Compare results with known analytical solutions
```

---

#### Module 3: matrix.py

**Priority**: HIGH - Used by many modules  
**Complexity**: Low  
**Estimated Time**: 2-3 hours  
**Dependencies**: constants.py  
**C Files**: `src/matrix.h` (18 lines) + `src/matrix.c` (76 lines)

**Translation Prompt**:
```
Translate src/matrix.h and src/matrix.c to wannsymm/matrix.py:

1. Wrapper functions around numpy/scipy for matrix operations
2. Replace MKL calls with numpy/scipy equivalents:
   - cblas_zgemm → np.matmul or @
   - LAPACKE_zgetrf, LAPACKE_zgetri → np.linalg.inv
   - LAPACKE_zheev → np.linalg.eigh

3. Functions to translate:
   - Matrix printing/display (pretty format)
   - Complex matrix multiplication
   - Matrix inversion
   - Hermitian matrix diagonalization
   - Any utility functions

4. Use numpy.ndarray as matrix type
5. Support both real and complex matrices
6. Add error checking (singular matrices, dimension mismatches)
7. Add type hints and docstrings
```

**Test Prompt**:
```
Create tests/test_matrix.py:

1. Test matrix multiplication (real and complex)
2. Test matrix inversion
3. Test eigenvalue decomposition
4. Test with known matrices (identity, diagonal, etc.)
5. Test error handling (singular, wrong dimensions)
6. Verify numerical accuracy
```

---

### Phase 2: Core Structures

#### Module 4: wannorb.py

**Priority**: HIGH - Core data structure  
**Complexity**: Low  
**Estimated Time**: 2-3 hours  
**Dependencies**: constants.py, vector.py  
**C Files**: `src/wannorb.h` (33 lines) + `src/wannorb.c` (36 lines)

**Translation Prompt**:
```
Translate src/wannorb.h and src/wannorb.c to wannsymm/wannorb.py:

1. Create WannOrb class (or dataclass):
   @dataclass
   class WannOrb:
       site: Vector          # orbital site position
       axis: np.ndarray      # 3x3 array or list of 3 Vectors
       r: int                # main quantum number
       l: int                # angular momentum (0=s, 1=p, 2=d, 3=f)
       mr: int               # cubic harmonic indicator
       ms: int               # spin: 0=up, 1=down

2. Implement initialization method init_wannorb()
3. Implement find_index_of_wannorb() function
4. Add validation (l >= 0, ms in [0,1], etc.)
5. Add __repr__ for readable display
6. Add type hints and docstrings

Orbital types:
- s: l=0 (no direction)
- p: l=1 (px, py, pz)
- d: l=2 (5 d-orbitals)
- f: l=3 (7 f-orbitals)
```

**Test Prompt**:
```
Create tests/test_wannorb.py:

1. Test WannOrb creation with various parameters
2. Test orbital properties access
3. Test finding orbitals in lists
4. Test different orbital types (s, p, d, f)
5. Test validation (invalid l, ms values)
6. Test equality comparison
```

---

#### Module 5: wanndata.py

**Priority**: HIGH - Main data structure  
**Complexity**: Medium  
**Estimated Time**: 6-8 hours  
**Dependencies**: constants.py, vector.py, wannorb.py  
**C Files**: `src/wanndata.h` (59 lines) + `src/wanndata.c` (284 lines)

**Translation Prompt**:
```
Translate src/wanndata.h and src/wanndata.c to wannsymm/wanndata.py:

1. Create WannData class:
   class WannData:
       norb: int                    # number of orbitals
       nrpt: int                    # number of R-points
       ham: np.ndarray              # complex array (Hamiltonian)
       hamflag: np.ndarray          # int array (flags)
       rvec: List[Vector]           # R-vectors
       weight: np.ndarray           # int array (weights)
       kvec: Optional[Vector]       # k-vector (for H(k))

2. Implement methods:
   - __init__(): Initialize structure
   - read_ham(filename): Read from _hr.dat file
   - write_ham(filename): Write to _hr.dat file
   - write_reduced_ham(filename): Write reduced version
   - find_index_of_ham(): Find matrix element index
   - __del__() or cleanup method

3. File format for _hr.dat (Wannier90 format):
   Line 1: Date/time comment
   Line 2: norb
   Line 3: nrpt
   Lines 4+: weights (15 per line)
   Remaining: R-vector and Hamiltonian elements
   Format: Rx Ry Rz  i j  Re(H) Im(H)

4. Implement additional structures:
   - HamBlock class (if needed)
   - HamR class for R-block format
   - Related conversion functions

5. Implement combine_wanndata() for spin combination
6. Add extensive error checking for file I/O
7. Use numpy for array operations
8. Add type hints and docstrings
```

**Test Prompt**:
```
Create tests/test_wanndata.py:

1. Test WannData initialization
2. Test reading _hr.dat files (create test files)
3. Test writing _hr.dat files
4. Test file format validation
5. Test finding Hamiltonian elements
6. Test with different sizes (small and large)
7. Test error handling (corrupt files, wrong format)
8. Test combine_wanndata() for spin channels
9. Verify round-trip (read-write-read consistency)
```

---

### Phase 3: I/O and Utilities

#### Module 6: usefulio.py

**Priority**: MEDIUM  
**Complexity**: Low  
**Estimated Time**: 2-3 hours  
**Dependencies**: Standard library  
**C Files**: `src/usefulio.h` (19 lines) + `src/usefulio.c` (55 lines)

**Translation Prompt**:
```
Translate src/usefulio.h and src/usefulio.c to wannsymm/usefulio.py:

1. Memory usage monitoring:
   - Use psutil for cross-platform memory monitoring
   - Function to report current memory usage
   - Function to report peak memory usage

2. Progress reporting:
   - Write to .progress-of-threadN files
   - Format similar to C version

3. System resource utilities:
   - CPU usage monitoring
   - Timing utilities

4. Use Python's logging module for output
5. Add type hints and docstrings
```

**Test Prompt**:
```
Create tests/test_usefulio.py:

1. Test memory usage reporting
2. Test progress file creation and writing
3. Test timing utilities
4. Test with actual memory allocation
```

---

#### Module 7: readinput.py

**Priority**: CRITICAL - Complex module  
**Complexity**: High  
**Estimated Time**: 20-30 hours  
**Dependencies**: constants.py, vector.py, wannorb.py, wanndata.py, matrix.py  
**C Files**: `src/readinput.h` (157 lines) + `src/readinput.c` (1,673 lines)

**Translation Prompt**:
```
Translate src/readinput.h and src/readinput.c to wannsymm/readinput.py:

This is the LARGEST and most complex module. Break into multiple steps:

STEP 1: Create InputData structure
   class InputData:
       # Required tags
       dftcode: str              # 'VASP' or 'QE'
       spinors: bool             # True for SOC
       seedname: str             # for _hr.dat files
       poscar_file: str          # structure file
       projections: List[...]    # orbital projections
       
       # Optional tags (with defaults)
       magmom: Optional[np.ndarray]
       symm_magnetic_tolerance: float = 1e-3
       global_trsymm: Optional[bool]
       kpt: Optional[Vector]
       degenerate_tolerance: float = 1e-6
       restart: bool = False
       use_symmetry: Optional[str]
       everysymm: bool = False
       expandrvec: bool = True
       ham_tolerance: float = 0.1
       kpath: Optional[List[...]]
       bands: Optional[bool]
       chaeig: Optional[bool]
       chaeig_in_kpath: bool = False
       saxis: np.ndarray = np.array([0, 0, 1])
       output_mem_usage: bool = False

STEP 2: Input file parsing
   - Read wannsymm.in file line by line
   - Handle comments (#, !, //)
   - Parse tag = value format
   - Parse begin/end blocks (projections, kpath, structure)
   - Case-insensitive tag names
   - Support both T/True and F/False for booleans

STEP 3: POSCAR reading
   - Parse VASP POSCAR/CONTCAR format
   - Extract lattice vectors (3x3 matrix)
   - Extract atomic positions
   - Extract atomic species and counts
   - Support both Direct and Cartesian coordinates
   - Consider using ASE library: from ase.io import read

STEP 4: Projections parsing
   - Format: "Element: orbitals"
   - Example: "Mn: s;d" or "F: px;py;pz"
   - Support shorthand: "d" → all 5 d-orbitals
   - Support specific: "dxy;dxz" → specific d-orbitals
   - Create WannOrb list from projections

STEP 5: K-path parsing
   - Format: "Label x y z Label x y z"
   - Example: "G 0 0 0 X 0.5 0 0"
   - Create list of k-point segments

STEP 6: Validation
   - Check required tags exist
   - Validate values (spinors is bool, etc.)
   - Set defaults for optional tags
   - Check file exists (POSCAR, use_symmetry)
   - Validate MAGMOM length matches atoms

STEP 7: Main function
   def read_input(filename='wannsymm.in') -> InputData:
       # Read and parse entire input file
       # Return populated InputData object

Implementation notes:
- Use configparser or custom parser
- Robust error messages for user
- Line number reporting for errors
- Support all tags from README.md
```

**Test Prompt**:
```
Create tests/test_readinput.py:

1. Create test input files for different scenarios:
   - Minimal required tags
   - All optional tags
   - With projections block
   - With kpath block
   - With structure block
   - SOC vs non-SOC
   - Magnetic vs non-magnetic

2. Test individual parsers:
   - Tag parsing
   - Boolean parsing (T/F, True/False)
   - Array parsing (MAGMOM)
   - Block parsing (projections, kpath)

3. Test POSCAR reading:
   - Simple structures
   - Different formats
   - Direct vs Cartesian

4. Test validation:
   - Missing required tags (should error)
   - Invalid values
   - File not found errors

5. Test defaults:
   - Optional tags use correct defaults

6. Test error messages are informative

7. Integration test: Read complete input file
```

---

#### Module 8: readsymm.py

**Priority**: HIGH  
**Complexity**: Low-Medium  
**Estimated Time**: 3-4 hours  
**Dependencies**: constants.py, vector.py, wannorb.py, readinput.py  
**C Files**: `src/readsymm.h` (13 lines) + `src/readsymm.c` (75 lines)

**Translation Prompt**:
```
Translate src/readsymm.h and src/readsymm.c to wannsymm/readsymm.py:

1. Use spglib Python bindings:
   import spglib

2. Implement read_symmetry_file():
   - Parse use_symmetry file format
   - Format:
     nsymm=N
     --- 1 ---
     r11 r12 r13
     r21 r22 r23
     r31 r32 r33
     tx ty tz T/F
   - Return list of (rotation, translation, time_reversal)

3. Implement automatic symmetry finding with spglib:
   - spglib.get_symmetry() for non-magnetic
   - spglib.get_magnetic_symmetry() for magnetic
   - Apply symm_tolerance

4. Handle time-reversal symmetry:
   - Based on Global_TRsymm tag
   - Or MAGMOM presence

5. Validate symmetry operations:
   - Check rotation matrices are orthogonal
   - Check det(rotation) = ±1

6. Return symmetry list in Symm format

Implementation notes:
- spglib uses different conventions, may need conversion
- Rotation matrices: spglib uses integers, ensure correct
- Translation: fractional coordinates
```

**Test Prompt**:
```
Create tests/test_readsymm.py:

1. Test parsing symmetry file
2. Test single symmetry operation parsing
3. Test time-reversal flag handling
4. Test spglib integration (simple structures)
5. Test symmetry validation
6. Test with known symmetries (cubic, etc.)
```

---

### Phase 4: Rotation Operations

#### Module 9: rotate_orbital.py

**Priority**: CRITICAL - Complex physics  
**Complexity**: High  
**Estimated Time**: 10-15 hours  
**Dependencies**: constants.py  
**C Files**: `src/rotate_orbital.h` (15 lines) + `src/rotate_orbital.c` (322 lines)

**Translation Prompt**:
```
Translate src/rotate_orbital.h and src/rotate_orbital.c to wannsymm/rotate_orbital.py:

This module implements quantum mechanical rotation of orbitals.

1. Implement Wigner D-matrices:
   - Use scipy.special for special functions
   - Or implement from scratch using Wigner formulas
   - Reference: Edmonds "Angular Momentum in Quantum Mechanics"

2. Orbital rotation for each l:
   - l=0 (s): Identity (no rotation)
   - l=1 (p): 3x3 rotation matrix
   - l=2 (d): 5x5 rotation matrix
   - l=3 (f): 7x7 rotation matrix

3. Spherical harmonics rotation:
   - Y_lm → Σ_m' D^l_mm'(R) Y_lm'
   - D^l_mm' are Wigner D-matrix elements

4. Cubic harmonics:
   - Real combinations of spherical harmonics
   - Transformation between spherical and cubic
   - Common in DFT codes

5. Key functions:
   def wigner_d_matrix(l, rotation):
       # Return (2l+1)x(2l+1) matrix
   
   def rotate_orbital(l, mr, rotation):
       # Rotate single orbital
       # Return rotation matrix
   
   def rotate_orbital_basis(orbitals, rotation):
       # Rotate entire basis
       # Return block-diagonal matrix

Implementation notes:
- Carefully follow C implementation
- Test against known rotations (90°, 180°, etc.)
- Consider caching matrices for performance
- May need numba for performance
```

**Test Prompt**:
```
Create tests/test_rotate_orbital.py:

1. Test Wigner D-matrix calculation
2. Test s-orbital (should be identity)
3. Test p-orbital rotation:
   - 90° around z: px → py, py → -px
   - 180° around z: px → -px, py → -py
4. Test d-orbital rotation (known cases)
5. Test orthogonality: D† D = I
6. Test unitarity: det(D) = 1
7. Test with random rotations
8. Compare with known tables of Wigner D-matrices
```

---

#### Module 10: rotate_spinor.py

**Priority**: HIGH  
**Complexity**: Medium  
**Estimated Time**: 4-6 hours  
**Dependencies**: constants.py  
**C Files**: `src/rotate_spinor.h` (9 lines) + `src/rotate_spinor.c` (65 lines)

**Translation Prompt**:
```
Translate src/rotate_spinor.h and src/rotate_spinor.c to wannsymm/rotate_spinor.py:

This module implements spin rotation for SOC calculations.

1. Pauli matrices:
   sigma_x = [[0, 1], [1, 0]]
   sigma_y = [[0, -1j], [1j, 0]]
   sigma_z = [[1, 0], [0, -1]]

2. Spin rotation matrix:
   R = exp(-i θ/2 n·σ)
   where n is rotation axis (unit vector)
         θ is rotation angle
         σ = (σ_x, σ_y, σ_z)

3. Implementation:
   def spin_rotation_matrix(rotation_matrix):
       # Convert 3x3 spatial rotation to 2x2 spin rotation
       # Extract axis and angle from rotation matrix
       # Compute exp(-i θ/2 n·σ)
       # Return 2x2 complex matrix

4. Euler angles or axis-angle:
   - May need to convert rotation matrix to axis-angle
   - Use scipy.spatial.transform.Rotation

5. Handle special cases:
   - Identity: angle = 0
   - 180° rotation: special formula
   - Inversion: different from rotation

Implementation notes:
- SU(2) group (2x2 unitary matrices)
- det(R) = 1 (special unitary)
- R† R = I (unitary)
- Carefully match C implementation
```

**Test Prompt**:
```
Create tests/test_rotate_spinor.py:

1. Test Pauli matrices properties
2. Test identity rotation (angle=0)
3. Test 180° rotation around z
4. Test 360° rotation (returns -I, not I)
5. Test unitarity: R† R = I
6. Test det(R) = 1
7. Test with known rotations
8. Compare with analytical formulas
```

---

#### Module 11: rotate_basis.py

**Priority**: HIGH  
**Complexity**: Medium  
**Estimated Time**: 6-8 hours  
**Dependencies**: vector.py, wanndata.py, rotate_orbital.py, rotate_spinor.py, rotate_ham.py  
**C Files**: `src/rotate_basis.h` (16 lines) + `src/rotate_basis.c` (140 lines)

**Translation Prompt**:
```
Translate src/rotate_basis.h and src/rotate_basis.c to wannsymm/rotate_basis.py:

This module combines orbital and spinor rotations.

1. For non-SOC case:
   - Only orbital rotation needed
   - Block diagonal matrix (one block per orbital)

2. For SOC case:
   - Tensor product of orbital and spin
   - |orbital, spin⟩ = |orbital⟩ ⊗ |spin⟩
   - Rotation: R_orb ⊗ R_spin

3. Implementation:
   def rotate_basis(wannorbs, rotation, spinors=False):
       if not spinors:
           # Non-SOC: block diagonal
           # Each block is rotate_orbital(l)
       else:
           # SOC: tensor product
           # Each block is kron(rotate_orbital(l), rotate_spinor())
       return rotation_matrix

4. Basis ordering:
   - Non-SOC: |orb1⟩, |orb2⟩, ..., |orbN⟩
   - SOC: |orb1,↑⟩, |orb1,↓⟩, |orb2,↑⟩, |orb2,↓⟩, ...
   - Match C code ordering exactly!

5. Handle site transformations:
   - If site moves under symmetry, adjust basis

Implementation notes:
- Use numpy.kron for tensor products
- Result should be norb×norb (non-SOC) or 2norb×2norb (SOC)
- Must be unitary: U† U = I
```

**Test Prompt**:
```
Create tests/test_rotate_basis.py:

1. Test non-SOC basis rotation
2. Test SOC basis rotation
3. Test unitarity: U† U = I
4. Test with simple orbital sets (all s, all p, etc.)
5. Test basis ordering matches expected
6. Test with known symmetries
```

---

#### Module 12: rotate_ham.py

**Priority**: CRITICAL - Core algorithm  
**Complexity**: High  
**Estimated Time**: 15-20 hours  
**Dependencies**: vector.py, matrix.py, wanndata.py, readinput.py, rotate_orbital.py, rotate_spinor.py  
**C Files**: `src/rotate_ham.h` (29 lines) + `src/rotate_ham.c` (861 lines)

**Translation Prompt**:
```
Translate src/rotate_ham.h and src/rotate_ham.c to wannsymm/rotate_ham.py:

This is the CORE symmetrization algorithm.

STEP 1: R-vector transformation
   def transform_rvector(rvec, rotation):
       # R' = rotation · R
       # Return transformed R-vector

STEP 2: Find transformed R-vector
   def find_transformed_rvec(rvec, rvec_list, tolerance):
       # Find rvec in list (with tolerance)
       # Return index or -1

STEP 3: Rotate Hamiltonian
   def rotate_hamiltonian(ham, rvec, rotation, wannorbs, spinors):
       # 1. Transform R-vectors: R' = rotation · R
       # 2. Build basis rotation matrix U
       # 3. Rotate H: H'(R') = U† H(R) U
       # Return rotated Hamiltonian

STEP 4: Symmetrization
   def symmetrize_hamiltonian(wann, symmetries, wannorbs, spinors, expandrvec):
       # For each symmetry S:
       #   H_S = rotate_hamiltonian(H, S)
       # H_symm = (1/N_symm) Σ_S H_S
       # Return symmetrized Hamiltonian

STEP 5: Expand R-vectors
   def expand_rvectors(wann, symmetries):
       # Find all R' = S · R for all S, R
       # Add missing R-vectors to list

STEP 6: Time-reversal symmetry
   def apply_time_reversal(ham):
       # H(-R)_ij = [H(R)_ji]*
       # Complex conjugate and transpose

STEP 7: Check consistency
   def check_hamiltonian_consistency(ham_orig, ham_symm, tolerance):
       # Compare on overlapping R-vectors
       # Warn if difference > tolerance

Main algorithm:
```python
def symmetrize(input_data):
    # 1. Read Hamiltonian
    wann = WannData()
    wann.read_ham(input_data.seedname)
    
    # 2. Expand R-vectors if requested
    if input_data.expandrvec:
        expand_rvectors(wann, symmetries)
    
    # 3. Initialize symmetrized Hamiltonian
    ham_symm = np.zeros_like(wann.ham)
    
    # 4. Loop over symmetries
    for sym in symmetries:
        ham_rotated = rotate_hamiltonian(wann, sym, ...)
        ham_symm += ham_rotated
    
    # 5. Average
    ham_symm /= len(symmetries)
    
    # 6. Apply time-reversal if needed
    if use_time_reversal:
        ham_tr = apply_time_reversal(ham_symm)
        ham_symm = (ham_symm + ham_tr) / 2
    
    # 7. Write output
    wann.ham = ham_symm
    wann.write_ham(input_data.seedname + '_symmed')
    
    return wann
```

Implementation notes:
- This is the computational bottleneck
- Use numpy vectorization aggressively
- Consider parallelization (multiprocessing or mpi4py)
- May need numba for hot loops
- Carefully handle R-vector indexing
- Match C implementation exactly for validation
```

**Test Prompt**:
```
Create tests/test_rotate_ham.py:

1. Test R-vector transformation
2. Test finding R-vectors in list
3. Test Hamiltonian rotation:
   - With identity symmetry (should be unchanged)
   - With simple rotations
4. Test time-reversal symmetry
5. Test Hermiticity: H(R)† = H(-R)
6. Test with small test Hamiltonian
7. Test consistency checking
8. Integration test with known system
```

---

### Phase 5: Analysis

#### Module 13: bndstruct.py

**Priority**: MEDIUM  
**Complexity**: Medium-High  
**Estimated Time**: 10-12 hours  
**Dependencies**: vector.py, matrix.py, wanndata.py, readinput.py, rotate_basis.py, rotate_orbital.py, rotate_spinor.py, rotate_ham.py  
**C Files**: `src/bndstruct.h` (49 lines) + `src/bndstruct.c` (352 lines)

**Translation Prompt**:
```
Translate src/bndstruct.h and src/bndstruct.c to wannsymm/bndstruct.py:

1. Fourier transform H(R) → H(k):
   def fourier_transform_ham(wann, kvec):
       # H(k) = Σ_R exp(ik·R) H(R)
       hamk = np.zeros((wann.norb, wann.norb), dtype=complex)
       for i, rvec in enumerate(wann.rvec):
           phase = np.exp(1j * 2 * np.pi * np.dot(kvec, rvec))
           hamk += phase * wann.ham[i] / wann.weight[i]
       return hamk

2. Band structure calculation:
   def calculate_band_structure(wann, kpath, npts=100):
       # Interpolate along k-path
       # For each k-point:
       #   Compute H(k)
       #   Diagonalize: eigenvalues = bands
       # Return bands(k)

3. Character calculation:
   def calculate_characters(wann, symmetries, kpt, tolerance):
       # For each symmetry S:
       #   Compute H(k)
       #   Diagonalize: eigenvectors
       #   Compute character: Tr[U†(S) P U(S)]
       # Return character table

4. Output functions:
   def write_band_structure(bands, filename):
       # Format: k_distance eigenvalues
   
   def write_characters(characters, filename):
       # Format: symmetry band_index character

Implementation notes:
- Use numpy.linalg.eigh for Hermitian matrices
- Handle degenerate eigenvalues carefully
- K-path interpolation: linear between points
- Efficient vectorization for many k-points
```

**Test Prompt**:
```
Create tests/test_bndstruct.py:

1. Test Fourier transform with simple H(R)
2. Test band structure calculation
3. Test with known tight-binding model
4. Test character calculation
5. Verify H(k) is Hermitian
6. Test eigenvalue ordering
7. Test output file format
```

---

### Phase 6: Main Program

#### Module 14: main.py

**Priority**: HIGH - Integration  
**Complexity**: Medium  
**Estimated Time**: 8-10 hours  
**Dependencies**: All modules  
**C Files**: `src/main.c` (789 lines)

**Translation Prompt**:
```
Translate src/main.c to wannsymm/main.py:

1. Command-line interface:
   import argparse
   
   def main():
       parser = argparse.ArgumentParser(
           description='WannSymm - Symmetry analysis for Wannier orbitals'
       )
       parser.add_argument('input', nargs='?', default='wannsymm.in',
                          help='Input file (default: wannsymm.in)')
       parser.add_argument('--version', action='version',
                          version=f'WannSymm {__version__}')
       args = parser.parse_args()
       
       run_wannsymm(args.input)

2. Main workflow:
   def run_wannsymm(input_file):
       # 1. Read input
       print("Reading input file...")
       input_data = read_input(input_file)
       
       # 2. Read structure
       print("Reading structure...")
       structure = read_poscar(input_data.poscar_file)
       
       # 3. Find symmetries
       print("Finding symmetries...")
       if input_data.use_symmetry:
           symmetries = read_symmetry_file(input_data.use_symmetry)
       else:
           symmetries = find_symmetries(structure, input_data)
       print(f"Found {len(symmetries)} symmetries")
       
       # 4. Read Hamiltonian
       print("Reading Hamiltonian...")
       wann = WannData()
       wann.read_ham(input_data.seedname)
       print(f"Hamiltonian: {wann.norb} orbitals, {wann.nrpt} R-points")
       
       # 5. Symmetrize (main task)
       if not input_data.restart:
           print("Symmetrizing Hamiltonian...")
           wann_symm = symmetrize_hamiltonian(
               wann, symmetries, input_data.wannorbs, input_data.spinors,
               input_data.expandrvec
           )
           
           # 6. Write output
           print("Writing symmetrized Hamiltonian...")
           wann_symm.write_ham(input_data.seedname + '_symmed')
       else:
           print("Restart mode: reading symmetrized Hamiltonian...")
           wann_symm = WannData()
           wann_symm.read_ham(input_data.seedname + '_symmed')
       
       # 7. Calculate bands if requested
       if input_data.bands and input_data.kpath:
           print("Calculating band structure...")
           bands_orig = calculate_band_structure(wann, input_data.kpath)
           bands_symm = calculate_band_structure(wann_symm, input_data.kpath)
           write_band_structure(bands_orig, 'bands_ori.dat')
           write_band_structure(bands_symm, 'bands_symmed.dat')
       
       # 8. Calculate characters if requested
       if input_data.chaeig and input_data.kpt:
           print("Calculating characters...")
           characters = calculate_characters(
               wann_symm, symmetries, input_data.kpt,
               input_data.degenerate_tolerance
           )
           write_characters(characters, 'bnd_sym_characters')
       
       print("Done!")

3. Error handling:
   - Wrap in try/except
   - Informative error messages
   - Clean up on error

4. Progress reporting:
   - Use logging module
   - Optional verbose mode
   - Progress bars for long operations (tqdm)

5. Optional MPI support:
   try:
       from mpi4py import MPI
       comm = MPI.COMM_WORLD
       rank = comm.Get_rank()
       size = comm.Get_size()
   except ImportError:
       rank = 0
       size = 1
```

**Test Prompt**:
```
Create tests/test_integration.py:

1. End-to-end test with small system
2. Test command-line interface
3. Test with different input options
4. Test restart mode
5. Test bands calculation
6. Test characters calculation
7. Compare with C version output (if available)
```

---

## Testing Strategy

### Unit Tests
- Test each module independently
- Use pytest framework
- Aim for >80% code coverage
- Test edge cases and error handling

### Integration Tests
- Test module interactions
- Use example systems from examples/
- Compare with C version output

### Validation Tests
- Use systems with known symmetries
- Check physical properties (Hermiticity, etc.)
- Numerical accuracy checks

### Regression Tests
- Keep test outputs for comparison
- Detect unintended changes

---

## Execution Plan

### Week 1: Foundation
- Day 1: constants.py + tests
- Day 2: vector.py + tests
- Day 3: matrix.py + wannorb.py + tests
- Day 4-5: wanndata.py + tests

### Week 2: I/O
- Day 1: usefulio.py + tests
- Day 2-4: readinput.py + tests (complex!)
- Day 5: readsymm.py + tests

### Week 3-4: Rotations
- Week 3 Day 1-2: rotate_orbital.py + tests
- Week 3 Day 3-4: rotate_spinor.py + tests
- Week 3 Day 5 - Week 4 Day 1: rotate_basis.py + tests
- Week 4 Day 2-5: rotate_ham.py + tests (complex!)

### Week 5: Analysis and Integration
- Day 1-2: bndstruct.py + tests
- Day 3-4: main.py + integration tests
- Day 5: Documentation and cleanup

### Week 6: Validation
- Compare with C version on all examples
- Performance profiling and optimization
- Bug fixes

---

## Success Criteria

For each module:
- ✅ All functions translated from C
- ✅ Type hints on all functions
- ✅ Docstrings in NumPy style
- ✅ Unit tests with >80% coverage
- ✅ All tests passing
- ✅ Code passes flake8 and mypy
- ✅ Formatted with black

For complete project:
- ✅ All modules complete
- ✅ Integration tests passing
- ✅ Matches C version output on examples
- ✅ Documentation complete
- ✅ Installation instructions work
- ✅ Ready for release

---

## Notes

- Keep C code as reference
- Match C behavior exactly for validation
- Optimize after correctness is verified
- Document differences from C version
- Consider performance from the start (use numpy efficiently)
