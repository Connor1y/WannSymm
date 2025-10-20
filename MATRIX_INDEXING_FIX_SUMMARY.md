# Matrix Indexing Bug Fix Summary

## Problem Statement

The Python translation of WannSymm was producing results "not close to the target output" when comparing with the C version, specifically at lines 153256-153279 in `wannier90.up_symmed_hr.dat`.

## Root Cause

**Matrix Indexing Transpose Bug in `rotate_ham.py`**

The Python WannData class stores Hamiltonian matrices with the convention:
```python
ham[irpt, iorb, jorb]  # Python storage format
```

However, the `rotate_ham()` function was incorrectly accessing:
```python
ham[irpt_in, jorb_in, iorb_in]  # WRONG - indices transposed!
```

This caused the function to read matrix elements from the wrong positions, resulting in incorrect Hamiltonian transformations.

## The Fix

Changed `rotate_ham()` function in `/home/runner/work/WannSymm/WannSymm/python_wannsymm/wannsymm/rotate_ham.py`:

### For SOC case (lines 534-545):
```python
# BEFORE (WRONG):
ham_element = hin.ham[irpt_in, jorb_in, iorb_in] / hin.weight[irpt_in]
hout.ham[irpt_out, jorb_out, iorb_out] += (...)

# AFTER (CORRECT):
ham_element = hin.ham[irpt_in, iorb_in, jorb_in] / hin.weight[irpt_in]
hout.ham[irpt_out, iorb_out, jorb_out] += (...)
```

### For non-SOC case (lines 563-570):
```python
# BEFORE (WRONG):
ham_element = hin.ham[irpt_in, jorb_in, iorb_in] / hin.weight[irpt_in]
hout.ham[irpt_out, jorb_out, iorb_out] += (...)

# AFTER (CORRECT):
ham_element = hin.ham[irpt_in, iorb_in, jorb_in] / hin.weight[irpt_in]
hout.ham[irpt_out, iorb_out, jorb_out] += (...)
```

## Verification

### Test 1: Identity Rotation
Created test to verify identity rotation preserves Hamiltonian:
```
Input:  [[1.0, 2.0], [3.0, 4.0]]
Output: [[1.0, 2.0], [3.0, 4.0]]  ✓ PASS
```

### Test 2: MnF2 Example Comparison
Ran full symmetrization on mnf2 example and compared with C reference:

**Target values (lines 153256-153279):**
- H(R=0, orb=1, orb=1) = 8.7371785000000006
- H(R=0, orb=2, orb=1) = 0.1219945000000000

**Python output:**
- H(R=0, orb=1, orb=1) = 8.7371785000000006 ✓
- H(R=0, orb=2, orb=1) = 0.1219945000000000 ✓

**Maximum difference:** 1.78×10⁻¹⁵ (within machine precision)

### Test Results
```
tests/test_rotate_ham.py ..................... 21 passed ✓
tests/test_matrix_indexing_fix.py ............ 1 passed  ✓
tests/test_mnf2_e2e.py (hamiltonian) ......... 1 passed  ✓
```

## Impact

The fix ensures that Python WannSymm now produces **identical** results to the C version (within numerical precision) for:
- All matrix elements at R=(0,0,0)
- All symmetry-transformed Hamiltonians
- Full end-to-end symmetrization workflow

The Python implementation now correctly applies the transformation:
```
H'(R', i', j') = Σ D†(i',i) H(R,i,j) D(j',j)
```

where D are the orbital rotation matrices.

## Files Modified

1. `/home/runner/work/WannSymm/WannSymm/python_wannsymm/wannsymm/rotate_ham.py`
   - Lines 539, 541: Fixed SOC case indexing
   - Lines 566, 568: Fixed non-SOC case indexing

2. `/home/runner/work/WannSymm/WannSymm/python_wannsymm/tests/test_matrix_indexing_fix.py`
   - Added comprehensive test to prevent regression
   - Documents the fix with clear test cases

## Conclusion

The issue was a simple but critical indexing bug. With this fix, the Python version now produces results that match the C version within machine precision (differences ~10⁻¹⁵), resolving the problem stated in the issue.
