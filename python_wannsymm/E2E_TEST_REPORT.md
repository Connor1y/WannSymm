# MnF2 End-to-End Test Report

## Test Summary

**Date**: 2025-10-20  
**Test Case**: MnF2 (anti-ferromagnetic MnF2 with collinear magnetism)  
**Python Implementation vs C Reference**

### Results Overview

| Test Case | Status | Details |
|-----------|--------|---------|
| test_hamiltonian_properties | ✅ PASS | Basic properties check on reference output |
| test_mnf2_up_spin | ❌ FAIL | R-point count mismatch (225 vs 533) |
| test_mnf2_dn_spin | ❌ FAIL | R-point count mismatch (225 vs 533) |

## Detailed Failure Analysis

### Primary Failure: Incomplete Hamiltonian Rotation

**Symptom**: Number of R-points mismatch
- Python output: 225 R-points
- C Reference: 533 R-points
- Max error: ~9.0 eV (consistency check failed)

**Root Cause**: `rotate_single_hamiltonian` function in `wannsymm/main.py` (lines 213-272) is a placeholder

### Module Fault Localization (Bottom-Up Dependency Analysis)

Based on the dependency chain and failure analysis:

```
Level 0 (Complete): constants, vector, matrix
Level 1 (Complete): wannorb, wanndata
Level 2 (Complete): usefulio, readinput, readsymm
Level 3 (Complete): rotate_orbital, rotate_spinor, rotate_basis
Level 4 (INCOMPLETE): rotate_ham - Missing implementation of rotate_single_hamiltonian
Level 5 (Affected): main workflow
```

**Smallest Responsible Module**: `rotate_ham` module, specifically the Hamiltonian rotation logic

### Failed Invariants

1. **R-vector Expansion**: When `expandrvec=True`, symmetry operations should generate new R-vectors through rotation S·R. The placeholder implementation doesn't do this.

2. **Hamiltonian Transformation**: For each symmetry operation, H'(R') = U† H(R) U where U is the orbital rotation matrix. This transformation is missing.

3. **Consistency**: The symmetrized Hamiltonian should match the original at the original R-points (within tolerance). Actual difference: 9.0 eV >> tolerance (0.1 eV).

## Issues Found and Fixed

### 1. MAGMOM Parsing Bug

**File**: `wannsymm/readinput.py`  
**Lines**: 938-942

**Problem**: Parser couldn't handle VASP notation like "4*0.0"

**Fix Applied**:
```python
def expand_vasp_notation(value_str: str) -> List[float]:
    """Expand VASP-style notation like "4*0.0" or "2*1.0 3*2.0"."""
    # Implementation added
```

**Status**: ✅ FIXED

### 2. Symmetry File Parsing Bug

**File**: `wannsymm/readsymm.py`  
**Lines**: 221-240

**Problem**: Parser failed on separator lines "--- N ---" in symmetries.dat

**Fix Applied**:
```python
# Skip separator lines like "--- N ---"
while line_idx < len(lines):
    line = lines[line_idx].strip()
    if line and not line.startswith('---'):
        break
    line_idx += 1
```

**Status**: ✅ FIXED

## Required Implementation for Full E2E Test Pass

### Main Missing Piece: rotate_single_hamiltonian

**Location**: `wannsymm/main.py`, lines 213-272

**Current State**: Placeholder that returns a copy of input

**Required Implementation**:

```python
def rotate_single_hamiltonian(
    ham_in: WannData,
    lattice: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    orb_info: List[WannOrb],
    flag_soc: int,
    flag_local_axis: int,
    index_of_sym: int
) -> WannData:
    """
    Apply symmetry operation to Hamiltonian.
    
    Algorithm (from C implementation):
    1. For each R-vector in input:
       - Rotate: R' = S·R (where S is rotation matrix)
       - Apply fractional translation
       - Map to output Hamiltonian
    
    2. For each orbital pair (i, j):
       - Get rotation matrices U_i, U_j for orbitals i and j
       - Transform: H'(R')[i',j'] = U_i† H(R)[i,j] U_j
       
    3. Handle SOC if flag_soc=1:
       - Apply spinor rotations
       - Use 2x2 spin rotation matrices
    
    4. Handle local axes if flag_local_axis=1:
       - Combine rotation with local orbital axes
    """
```

**Dependencies** (all implemented):
- `rotate_orbital.py`: `get_rotation_matrix()`, `rotate_Ylm()`, `rotate_cubic()`
- `rotate_spinor.py`: `rotate_spinor()`
- `rotate_basis.py`: `get_basis_rotation()`, `combine_basis_rotation()`
- `vector.py`: `vector_rotate()`, `find_vector()`
- `matrix.py`: matrix operations

**Reference**: `src/rotate_ham.c`, function `rotate_ham()`, lines 9-410

**Complexity**: ~400 lines of C code to translate

## Minimal Fix Proposal

### Option 1: Complete Translation (Recommended for Production)

Translate the C `rotate_ham` function to Python following the existing module pattern.

**Effort**: ~8-16 hours (complex logic, orbital/spinor rotations, debugging)

**Files to modify**:
1. `wannsymm/main.py` - Replace placeholder with full implementation
2. May need helper functions in `wannsymm/rotate_ham.py`

### Option 2: Simplified Test (Immediate)

Create a test that validates partial functionality:
- Input parsing ✅
- Symmetry reading ✅  
- Basic Hermiticity ✅
- Document rotation as TODO

**Status**: Already implemented in current test

## Test Execution Details

### Environment
- Python: 3.12.3
- NumPy: 2.3.4
- SciPy: 1.16.2
- spglib: 2.6.0
- ASE: 3.26.0

### Test Files
- Input: `examples/mnf2/input/wannier90.up_hr.dat` (129,618 lines, 24 orbitals, 225 R-points)
- Reference: `examples/mnf2/output/wannier90.up_symmed_hr.dat` (307,047 lines, 24 orbitals, 533 R-points)
- Symmetries: 16 magnetic symmetry operations

### Performance
- Parsing: ~0.5s
- Hermitian symmetry: ~0.1s  
- Placeholder rotation: ~1.5s (16 operations)
- Total test time: ~2-4s per spin channel

## Recommendations

### Immediate Actions

1. **Document Limitation**: Update README and documentation to note that Hamiltonian rotation is incomplete
2. **Mark as WIP**: Add clear markers in code about placeholder status
3. **Integration Test**: Keep current test as regression check for completed modules

### Next Steps

1. **Implement `rotate_ham`**: 
   - Start with non-SOC case (simpler)
   - Add SOC support
   - Add local axis support
   - Extensive unit testing

2. **Validate Against C Reference**:
   - Compare intermediate results
   - Check R-vector generation
   - Verify orbital rotations
   - Test edge cases

3. **Performance Optimization**:
   - Profile bottlenecks
   - Consider Numba JIT for hot loops
   - Vectorize where possible

## Conclusion

The Python translation is **~90% complete** in terms of module coverage, but missing the **critical rotate_ham implementation** which is the core of the symmetrization algorithm.

**Modules Complete**:
- ✅ constants, vector, matrix
- ✅ wannorb, wanndata
- ✅ usefulio, readinput, readsymm
- ✅ rotate_orbital, rotate_spinor, rotate_basis
- ✅ bndstruct (band structure calculation)
- ⚠️ rotate_ham (has helper functions but missing main rotation)
- ⚠️ main (workflow complete but uses placeholder rotation)

**Current Test Value**: 
- Validates I/O, parsing, and data structures
- Identifies the specific missing implementation
- Provides framework for validation once rotation is complete

**Next Milestone**: Complete `rotate_single_hamiltonian` to enable full E2E validation.
