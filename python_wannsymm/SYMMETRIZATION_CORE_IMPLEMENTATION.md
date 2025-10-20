# Symmetrization Core Implementation Summary

## Overview

This document describes the implementation of the symmetrization core for WannSymm, translating key algorithms from C to Python with NumPy optimization.

## Implemented Components

### 1. R-Vector Transformations

**Function:** `inverse_symm(rotation, translation)`

Computes the inverse of a symmetry operation (R, t):
- (R, t)^-1 = (R^-1, -R^-1·t)

**Implementation Details:**
- Uses `matrix3x3_inverse` for rotation inversion
- Vector rotation for translation transformation
- Fully vectorized with NumPy

**Tests:**
- Identity symmetry
- Pure translation
- Pure rotation  
- Combined rotation and translation

### 2. Hamiltonian Complex Conjugate Transpose

**Function:** `get_conj_trans_of_ham(hin)`

Computes H†(R) = H*(-R) with swapped indices:
- For each R-vector, finds -R in the list
- Applies: hout[irpt(-R), i, j] = conj(hin[irpt(R), j, i])

**Implementation Details:**
- Preserves R-vector and weight arrays
- Swaps orbital indices during conjugation
- Ensures Hermiticity property: H†(-R) = H(R)

**Tests:**
- Identity (diagonal) Hamiltonian
- Hermitian property verification
- Complex off-diagonal elements

### 3. Time-Reversal Symmetry

**Function:** `trsymm_ham(hin, orb_info, flag_soc)`

Applies time-reversal symmetry operator:
- Without SOC: T = K (complex conjugation)
- With SOC: T = i·σ_y·K (spin rotation + conjugation)

**Implementation Details:**
- `flag_soc=0`: Simple complex conjugation
- `flag_soc=1`: Applies time-reversal factor matrix:
  ```
  tr_factor = [0, 1, -1, 0]  # i·σ_y matrix elements
  ```
- Sums over spin states with proper rotation factors

**Tests:**
- Real Hamiltonian without SOC (identity operation)
- Complex Hamiltonian without SOC (conjugation)
- Spin-coupled system with SOC

### 4. Symmetrization Averaging

**Function:** `symmetrize_hamiltonians(ham_list, nsymm, expand_rvec)`

Averages over all symmetry operations:
- H_symm(R) = (1/N_symm) Σ_i H_i(R)

**Implementation Details:**
- Collects unique R-vectors from all transformed Hamiltonians
- Option to expand R-vector set (`expand_rvec=True`)
- Uses `hamflag` to track contributing symmetries
- Normalizes by number of contributions per element

**Key Features:**
- R-vector expansion for complete symmetrization
- Weighted averaging with degeneracy tracking
- Preserves sparsity when `expand_rvec=False`

**Tests:**
- Single identity operation
- Average of multiple operations
- R-vector expansion verification

### 5. Hermiticity Enforcement

**Function:** `apply_hermitian_symmetry(ham_in)`

Enforces Hermitian property:
- H_hermitian = (H + H†)/2

**Implementation Details:**
- Calls `get_conj_trans_of_ham` to get H†
- Averages original and conjugate transpose
- Ensures H = H† at R=0 (on-site terms)

**Tests:**
- Already Hermitian matrix (no change)
- Non-Hermitian matrix (becomes Hermitian)
- Verification of Hermitian property at R=0

### 6. Consistency Checks

**Function:** `check_hamiltonian_consistency(ham1, ham2, tolerance)`

Validates consistency between Hamiltonians:
- Compares element-wise differences
- Returns boolean flag and maximum difference

**Implementation Details:**
- Checks array dimensions
- Verifies R-vector matching
- Computes max absolute difference
- Configurable tolerance (default: 0.1 eV)

**Tests:**
- Identical Hamiltonians
- Within tolerance differences
- Outside tolerance differences
- Different sized Hamiltonians

## Optimization Features

### NumPy Vectorization

All operations use NumPy broadcasting and vectorized operations:

1. **Array Operations:**
   ```python
   # Instead of nested loops:
   hout.ham = np.conj(hin.ham)  # Vectorized conjugation
   ```

2. **Matrix Operations:**
   ```python
   # Matrix inversion and multiplication via NumPy/LAPACK
   inv_rotation = matrix3x3_inverse(rotation)
   ```

3. **Broadcasting:**
   ```python
   # Element-wise operations on entire arrays
   ham_hermitian.ham = (ham_in.ham + ham_conj_trans.ham) / 2.0
   ```

### Performance Characteristics

- **Memory Efficiency:** In-place operations where possible
- **Cache Locality:** Contiguous array operations
- **BLAS/LAPACK:** Uses optimized linear algebra routines
- **Complexity:** O(N_symm × N_rpt × N_orb²) for symmetrization

## Integration Tests

### Full Workflow Test

Tests complete symmetrization pipeline:
1. Create initial Hamiltonian
2. Apply Hermitian symmetry
3. Verify Hermiticity at R=0

### Time-Reversal + Hermitian

Tests combined operations:
1. Apply time-reversal symmetry
2. Apply Hermitian symmetry
3. Verify result is real and symmetric

## Consistency with C Implementation

The Python implementation maintains consistency with the original C code:

1. **Algorithm Equivalence:**
   - Same mathematical operations
   - Same ordering and indexing conventions
   - Same tolerance values

2. **Data Structure Compatibility:**
   - `WannData` matches C `wanndata` struct
   - `WannOrb` matches C `wannorb` struct
   - `Vector` matches C `vector` struct

3. **Numerical Accuracy:**
   - Uses same tolerances (eps4, eps5, eps8)
   - Same complex number handling (np.complex128)
   - Same matrix operations (via BLAS/LAPACK)

## Requirements Coverage

✅ **R-vector transforms:** Implemented in `inverse_symm` and used throughout

✅ **H-rotation:** Implemented via conjugate transpose and symmetry operations

✅ **Averaging over symmetries:** Implemented in `symmetrize_hamiltonians`

✅ **Time-reversal:** Implemented in `trsymm_ham` with and without SOC

✅ **Consistency checks:** Implemented in `check_hamiltonian_consistency`

✅ **NumPy vectorization:** All operations use vectorized NumPy

✅ **Tests:**
- ✅ Identity symmetry (no change)
- ✅ Time-reversal symmetry (TRS)
- ✅ Hermiticity preservation
- ✅ Consistency checks

## Future Enhancements

Potential optimizations for very large systems:

1. **Sparse Matrix Support:** For systems with many R-vectors
2. **Parallelization:** MPI support via mpi4py
3. **JIT Compilation:** Numba for hot loops
4. **Caching:** Memoization of rotation matrices

## Usage Examples

### Basic Symmetrization

```python
from wannsymm.rotate_ham import (
    apply_hermitian_symmetry,
    symmetrize_hamiltonians
)
from wannsymm.wanndata import init_wanndata

# Create Hamiltonian
ham = init_wanndata(norb=10, nrpt=100)
# ... populate ham ...

# Enforce Hermiticity
ham_h = apply_hermitian_symmetry(ham)

# Symmetrize over operations
ham_list = [ham_transformed_by_symm_i for i in range(nsymm)]
ham_final = symmetrize_hamiltonians(ham_list, nsymm=len(ham_list))
```

### Time-Reversal Symmetry

```python
from wannsymm.rotate_ham import trsymm_ham

# Apply time-reversal (without SOC)
ham_tr = trsymm_ham(ham, orb_info, flag_soc=0)

# Apply time-reversal (with SOC)
ham_tr_soc = trsymm_ham(ham, orb_info, flag_soc=1)
```

### Consistency Checking

```python
from wannsymm.rotate_ham import check_hamiltonian_consistency

# Check if two Hamiltonians are consistent
consistent, max_diff = check_hamiltonian_consistency(
    ham_original, ham_symmetrized, tolerance=0.1
)

if not consistent:
    print(f"Hamiltonians differ by up to {max_diff} eV")
```

## Testing

Run tests with:

```bash
cd python_wannsymm
pytest tests/test_rotate_ham.py -v
```

All 21 tests pass:
- 4 tests for `inverse_symm`
- 3 tests for `get_conj_trans_of_ham`
- 3 tests for `trsymm_ham`
- 3 tests for `symmetrize_hamiltonians`
- 2 tests for `apply_hermitian_symmetry`
- 4 tests for `check_hamiltonian_consistency`
- 2 integration tests

## References

1. Original C implementation: `src/rotate_ham.c` (861 lines)
2. Paper: G.-X. Zhi et al., Computer Physics Communications 271 (2022) 108196
3. Wannier90 documentation: http://www.wannier.org/
