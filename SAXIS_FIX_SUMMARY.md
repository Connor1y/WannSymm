# SAXIS Transformation Bug Fix

## Summary

Fixed a critical bug in the SAXIS (spin quantization axis) transformation that affects magnetic materials with non-default spin quantization axes. The bug was present in both the C code (`src/readinput.c`) and was missing entirely from the Python implementation.

## The Bug

### C Code Bug (Line 1316 in `src/readinput.c`)

The rotation matrix formula for transforming magnetic moments from the spin quantization axis frame to Cartesian coordinates had an incorrect term:

```c
// BEFORE (BUGGY):
magmom[ii].y = cos(beta)*sin(alpha)*maxis.x + cos(alpha)*maxis.y + sin(beta)*cos(alpha)*maxis.z;
//                                                                   ^^^^^^^^^^^^^^^^^ BUG!

// AFTER (FIXED):
magmom[ii].y = cos(beta)*sin(alpha)*maxis.x + cos(alpha)*maxis.y + sin(beta)*sin(alpha)*maxis.z;
//                                                                   ^^^^^^^^^^^^^^^^^^ CORRECT!
```

The term `sin(beta)*cos(alpha)` should have been `sin(beta)*sin(alpha)`.

### Missing Python Implementation

The Python code was completely missing the SAXIS transformation. When reading magmom values from input files, it simply treated 1D arrays as z-components without applying the proper rotation.

## Impact

### Materials Affected

**NOT affected (default SAXIS):**
- MnF2 example (uses default SAXIS = (0, 0, 1))
- Most collinear magnetic systems that use default settings
- When SAXIS = (0, 0, 1): alpha=0, beta=0, so sin(beta)=0 and the bug doesn't manifest

**AFFECTED (non-default SAXIS):**
- Materials with spin quantization axis not along z (e.g., SAXIS = (1, 0, 0) or (1, 1, 0))
- Non-collinear magnetic systems
- SOC calculations with specific spin orientation

### Example of Bug Impact

For SAXIS = (1, 0, 0) with magmom = [4.6] (single atom):
- **Expected (correct physics):** magmom_cartesian = (4.6, 0, 0)
- **Buggy C code produced:** magmom_cartesian = (4.6, 4.6, 0) - spurious y-component!
- **Python before fix:** magmom_cartesian = (0, 0, 4.6) - no transformation at all!

## The Fix

### 1. Added SAXIS Transformation to Python

**File:** `python_wannsymm/wannsymm/readinput.py`

Added new function `apply_saxis_transformation()` that:
- Calculates rotation angles from SAXIS vector
- Applies proper rotation matrix to convert magmom to Cartesian coordinates
- Handles both SOC and non-SOC cases
- Uses the **corrected** rotation formula

```python
def apply_saxis_transformation(
    magmom_array: List[float],
    natom: int,
    flag_soc: int,
    saxis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> npt.NDArray[np.float64]:
    """Apply SAXIS transformation to convert magmom to Cartesian coordinates."""
    # Calculate angles
    alpha = ... # azimuthal angle
    beta = ...  # polar angle
    
    # Apply correct rotation matrix
    magmom[ii, 0] = cos(beta)*cos(alpha)*maxis_x - sin(alpha)*maxis_y + sin(beta)*cos(alpha)*maxis_z
    magmom[ii, 1] = cos(beta)*sin(alpha)*maxis_x + cos(alpha)*maxis_y + sin(beta)*sin(alpha)*maxis_z  # FIXED
    magmom[ii, 2] = -sin(beta)*maxis_x + cos(beta)*maxis_z
```

### 2. Integrated into Main Workflow

**File:** `python_wannsymm/wannsymm/main.py`

Applied transformation before passing magmom to symmetry detection:

```python
# Process magmom with SAXIS transformation if needed
processed_magmom = input_data.magmom
if input_data.magmom is not None:
    from .readinput import apply_saxis_transformation
    natom = sum(input_data.num_atoms_each)
    flag_soc = 1 if input_data.spinors else 0
    processed_magmom = apply_saxis_transformation(
        input_data.magmom,
        natom,
        flag_soc,
        input_data.saxis
    )

symm_data = find_symmetries_with_spglib(
    ...
    magmom=processed_magmom,  # Use transformed magmom
    ...
)
```

### 3. Fixed C Code

**File:** `src/readinput.c` (Line 1316)

Changed:
```c
sin(beta)*cos(alpha)*maxis.z  // WRONG
```
to:
```c
sin(beta)*sin(alpha)*maxis.z  // CORRECT
```

## Mathematics

The rotation that takes the z-axis to SAXIS direction is:
1. Rotate around z-axis by angle `alpha` (azimuthal)
2. Rotate around new y-axis by angle `beta` (polar)

The combined rotation matrix is:
```
R = Ry(beta) * Rz(alpha)
  = [cos(β)cos(α),  -sin(α),  sin(β)cos(α)]
    [cos(β)sin(α),   cos(α),  sin(β)sin(α)]  <- Note the sin(α) term
    [  -sin(β),        0,       cos(β)    ]
```

Where:
- `alpha = atan(SAXIS.y / SAXIS.x)` (azimuthal angle in xy-plane)
- `beta = atan(sqrt(SAXIS.x^2 + SAXIS.y^2) / SAXIS.z)` (polar angle from z-axis)

## Testing

Added comprehensive test suite `tests/test_saxis_transformation.py`:
- Default SAXIS (identity transformation)
- SAXIS along x, y axes
- SAXIS in xy-plane (1,1,0)
- SOC mode with 3-component magmom
- Edge cases (near-zero components)

All tests pass with the corrected formula.

## Note on Line 46 Issue

The Line 46 issue in MnF2 (H(R=(-3,-3,-5), 7, 1) = -0.017184 instead of 0.0) is **NOT fixed** by this SAXIS correction because:
1. MnF2 uses default SAXIS = (0, 0, 1)
2. When beta=0, sin(beta)=0, so the buggy term evaluates to zero anyway
3. The Line 46 issue is related to orbital rotation or Hamiltonian transformation logic, not SAXIS

The Line 46 issue requires separate investigation.

## Files Modified

1. `src/readinput.c` - Fixed rotation formula (1 line)
2. `python_wannsymm/wannsymm/readinput.py` - Added transformation function
3. `python_wannsymm/wannsymm/main.py` - Integrated transformation
4. `python_wannsymm/wannsymm/readsymm.py` - Updated to accept transformed magmom
5. `python_wannsymm/tests/test_saxis_transformation.py` - New test suite

## Verification

- ✓ All existing tests pass
- ✓ New SAXIS transformation tests pass
- ✓ Magmom handling tests pass
- ✓ C and Python implementations now consistent
- ⚠ Line 46 issue still exists (separate problem)
