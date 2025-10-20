# Line 46 Issue Investigation Summary

## Problem
Line 46 in `wannier90.up_symmed_hr.dat` should have value 0.0 (as in C reference) but Python produces -0.0171840.

This represents: H(R=(-3,-3,-5), orbital_i=7, orbital_j=1)
- Orbital 1: Mn1 s-orbital (Mn at origin, magmom=+4.6)
- Orbital 7: Mn2 s-orbital (Mn at (0.5,0.5,0.5), magmom=-4.6)

## Key Findings

### 1. Reference Output Pattern at R=(-3,-3,-5)
The C reference has a SELECTIVE pattern of zeros for Mn1-Mn2 couplings:
```
Mn1→Mn2 at R=(-3,-3,-5):
  (1,7): NON-ZERO   (1,8): NON-ZERO   (1,9): ZERO   (1,10): ZERO   (1,11): ZERO   (1,12): NON-ZERO
  (2,7): NON-ZERO   (2,8): NON-ZERO   (2,9): ZERO   (2,10): ZERO   (2,11): ZERO   (2,12): NON-ZERO
  (3,7): ZERO       (3,8): ZERO       (3,9): NON-ZERO ...
  ... pattern suggests orbital angular momentum selection rules
```

This indicates that certain orbital pairs are FORBIDDEN by symmetry at this specific R-vector, not all Mn1-Mn2 pairs.

### 2. The Issue is NOT:
- ❌ Time-reversal handling (TR logic is correct)
- ❌ Averaging logic (matches C code exactly)
- ❌ Hermiticity enforcement (correct)
- ❌ Matrix indexing (fixed in previous PR)
- ❌ Input data (has non-zero Mn1-Mn2 couplings as expected)

### 3. The Issue IS Likely:
One of these in the `rotate_ham` function:

**Option A: Orbital Rotation Matrix Application**
The transformation H'(i',j') = Σ D†(i',i) H(i,j) D(j',j) might have:
- Wrong signs or conjugation
- Wrong matrix indexing for D
- Accumulated numerical errors breaking selection rules

**Option B: R-vector Rotation Logic**  
When rotating R-vectors and determining which input R contributes to which output R, 
there might be cases where:
- Incompatible R-vectors are being included
- The quasi-crystal check is insufficient
- Rounding of R-vectors introduces spurious contributions

**Option C: Find Orbital Index Logic**
When `find_index_of_wannorb` searches for the rotated orbital position, it might:
- Match orbitals that shouldn't match due to numerical tolerance
- Return wrong indices in edge cases

## Specific Code Locations to Investigate

### Primary Suspect: rotate_ham.py lines 514-572
The core rotation loop where orbital matrices are applied:
```python
for mr1 in range(1, 2*l1+2):
    for mr2 in range(1, 2*l2+2):
        iorb_in = find_index_of_wannorb(orb_info, site_in1, r1, l1, mr1, ms1)
        jorb_in = find_index_of_wannorb(orb_info, site_in2, r2, l2, mr2, ms2)
        
        orb_factor_left = orb_rot[l1][mr_i-1, mr1-1]
        orb_factor_right = np.conj(orb_rot[l2][mr_j-1, mr2-1])
        
        ham_element = hin.ham[irpt_in, iorb_in, jorb_in] / hin.weight[irpt_in]
        
        hout.ham[irpt_out, iorb_out, jorb_out] += (
            orb_factor_left * ham_element * orb_factor_right
        )
```

### Secondary Suspects:
1. **rotate_cubic** in `rotate_orbital.py` - generates orbital rotation matrices
2. **get_axis_angle_of_rotation** in `rotate_basis.py` - extracts rotation axis/angle
3. **getrvec_and_site** in `rotate_ham.py` - maps rotated positions to R+site

## Debugging Strategy

1. **Compare rotation matrices**: Check if `rotate_cubic` produces same matrices as C `rotate_orbital_functions`
2. **Trace single symmetry**: Follow symmetry operation #3 or #7 (which have (0.5,0.5,0.5) translation) through the entire rotation
3. **Check selection rules**: Verify that orbital rotation matrices naturally produce zeros for forbidden couplings
4. **Numerical precision**: Check if small numerical errors are accumulating and breaking exact zeros

## Recommended Fix Approach

1. Add detailed logging to `rotate_ham` to see which symmetries contribute to line 46 element
2. Compare C and Python rotation matrices element-by-element for a specific symmetry
3. Check if the issue is systematic (all similar elements wrong) or isolated (just this one)
4. Consider adding a post-processing step to enforce known selection rules based on orbital angular momentum

## Test Added
Created `test_line46_issue.py` which explicitly checks that H(R=(-3,-3,-5), 7, 1) ≈ 0.0
This test currently FAILS, capturing the bug for regression testing after fix.
