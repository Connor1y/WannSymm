# Quick Start Guide for Python Translation

This guide will help you get started translating WannSymm from C to Python.

## Prerequisites

- Python 3.7 or higher
- Understanding of C programming
- Basic quantum mechanics knowledge (helpful but not required for early modules)
- Git for version control

## Setup Development Environment

1. **Clone the repository** (already done)
   ```bash
   cd /path/to/WannSymm
   ```

2. **Create Python virtual environment**
   ```bash
   cd python_wannsymm
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Verify setup**
   ```bash
   python -c "import numpy; import scipy; import spglib; print('All dependencies OK')"
   pytest tests/  # Should run placeholder tests
   ```

## Translation Workflow for Each Module

### Step 1: Read the C Code
1. Open the C source file(s) in `../src/`
2. Understand the data structures (structs)
3. Understand the functions and algorithms
4. Note any external dependencies (MKL, MPI, etc.)

### Step 2: Read the Translation Prompt
1. Open `docs/TRANSLATION_WORKFLOW.md`
2. Find your module's section
3. Read the translation notes carefully
4. Understand the Python equivalent structures

### Step 3: Implement the Module
1. Open `wannsymm/MODULE_NAME.py`
2. Remove the TODO comments
3. Implement the functionality following the prompt
4. Add type hints to all functions
5. Add docstrings in NumPy style

Example structure:
```python
"""
Module description

Translated from: src/MODULE_NAME.c
"""

import numpy as np
from typing import Optional, List, Tuple

from .constants import *
from .vector import Vector


def function_name(param1: int, param2: np.ndarray) -> np.ndarray:
    """
    Brief description.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : np.ndarray
        Description of param2

    Returns
    -------
    np.ndarray
        Description of return value

    Examples
    --------
    >>> result = function_name(5, np.array([1, 2, 3]))
    >>> print(result)
    array([...])
    """
    # Implementation
    pass
```

### Step 4: Write Tests
1. Open `tests/test_MODULE_NAME.py`
2. Remove placeholder test
3. Write comprehensive tests:
   - Happy path cases
   - Edge cases
   - Error cases
   - Known analytical results

Example test structure:
```python
import pytest
import numpy as np
from wannsymm.MODULE_NAME import function_name


class TestFunctionName:
    def test_basic_case(self):
        """Test basic functionality"""
        result = function_name(5, np.array([1, 2, 3]))
        expected = np.array([...])
        np.testing.assert_array_almost_equal(result, expected)

    def test_edge_case(self):
        """Test edge case"""
        # ...

    def test_error_handling(self):
        """Test that errors are raised correctly"""
        with pytest.raises(ValueError):
            function_name(-1, np.array([]))
```

### Step 5: Run Tests
```bash
pytest tests/test_MODULE_NAME.py -v
```

Fix any failures and repeat until all tests pass.

### Step 6: Check Code Quality
```bash
# Format code
black wannsymm/MODULE_NAME.py

# Check style
flake8 wannsymm/MODULE_NAME.py

# Type checking
mypy wannsymm/MODULE_NAME.py
```

### Step 7: Update Progress
Update `README.md` status table:
- Change ⏳ to 🚧 when starting
- Change 🚧 to ✅ when complete

## Translation Best Practices

### 1. Stay Close to C Code
- Keep similar function names
- Keep similar algorithm structure
- Makes validation easier

### 2. Use NumPy Effectively
```python
# Good: Vectorized
result = np.sum(array * weights)

# Avoid: Loops where vectorization possible
result = 0
for i in range(len(array)):
    result += array[i] * weights[i]
```

### 3. Handle Complex Numbers
```python
# Use numpy complex128
z = np.complex128(1.0 + 2.0j)

# Use 1j for imaginary unit (not I or i)
phase = np.exp(1j * 2 * np.pi * k_dot_r)
```

### 4. Memory Management
Python handles memory automatically, but:
- Use `np.zeros()` for pre-allocation
- Don't worry about `free()` - Python handles it
- Use `del` for large arrays when done (optional)

### 5. Error Handling
```python
# Use exceptions instead of return codes
def read_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    # ...
```

### 6. Type Hints
```python
from typing import List, Optional, Tuple
import numpy as np

def process_data(
    data: np.ndarray,
    indices: List[int],
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, int]:
    """..."""
    # Implementation
    return result, count
```

## Common C to Python Mappings

### Data Types
- `int` → `int`
- `double` → `float`
- `double complex` → `np.complex128`
- `double *` → `np.ndarray`
- `struct` → `@dataclass` or `class`

### Memory
- `malloc()` → `np.zeros()` or `np.empty()`
- `free()` → (automatic)
- `memcpy()` → `array.copy()` or slicing

### Math Functions
- `sin()` → `np.sin()`
- `cos()` → `np.cos()`
- `exp()` → `np.exp()`
- `sqrt()` → `np.sqrt()`
- `fabs()` → `np.abs()` or `abs()`

### Linear Algebra (MKL → NumPy/SciPy)
- `cblas_zgemm()` → `np.matmul()` or `@`
- `LAPACKE_zgetrf/zgetri()` → `np.linalg.inv()`
- `LAPACKE_zheev()` → `np.linalg.eigh()`

### Arrays
```c
// C: Dynamic allocation
double *arr = malloc(n * sizeof(double));
for (int i = 0; i < n; i++) {
    arr[i] = i * 2.0;
}
```

```python
# Python: NumPy array
arr = np.arange(n) * 2.0
# or
arr = np.array([i * 2.0 for i in range(n)])
```

### Loops
```c
// C: Loop
for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
}
```

```python
# Python: Vectorized (preferred)
result = a + b

# Or if loop needed
for i in range(n):
    result[i] = a[i] + b[i]
```

## Module-Specific Tips

### constants.py
- Just define variables, very simple
- Use `import numpy as np` and define `COMPLEX_TYPE = np.complex128`

### vector.py
- Use `@dataclass` for Vector class
- Implement `__add__`, `__sub__`, `__mul__` operators
- Use `np.linalg.norm()` for vector norm

### matrix.py
- Thin wrapper around numpy/scipy
- Main purpose: consistent interface

### wannorb.py
- Similar to vector.py
- Dataclass with validation in `__post_init__`

### wanndata.py
- Most complex file I/O
- Study Wannier90 _hr.dat format carefully
- Read C code's file reading section carefully

### readinput.py
- Largest module - take your time
- Break into smaller functions
- Test each tag parser separately

### rotate_orbital.py, rotate_spinor.py
- These require physics knowledge
- Study references on Wigner D-matrices
- May need scipy.special functions
- Verify against known results

### rotate_ham.py
- Core algorithm - most important
- Performance critical - use numpy efficiently
- Carefully match C implementation

## Testing Strategies

### Unit Tests
- Test each function independently
- Use small, simple inputs
- Compare with analytical results

### Property Tests
- Check physical properties:
  - Hermiticity: H† = H
  - Unitarity: U† U = I
  - Orthogonality: ⟨i|j⟩ = δᵢⱼ

### Integration Tests
- Test module combinations
- Use small complete examples

### Validation Tests
- Compare with C version output
- Use examples from `../examples/`

## Debugging Tips

### Print Shapes
```python
print(f"array shape: {array.shape}, dtype: {array.dtype}")
```

### Check NaN/Inf
```python
assert not np.isnan(array).any(), "Found NaN values"
assert not np.isinf(array).any(), "Found Inf values"
```

### Visual Debugging
```python
import matplotlib.pyplot as plt
plt.imshow(np.abs(matrix))
plt.colorbar()
plt.show()
```

### Compare with C
```python
# Save intermediate results in both C and Python
# Compare them:
c_result = np.loadtxt('c_output.txt')
py_result = your_function()
np.testing.assert_allclose(py_result, c_result, rtol=1e-6)
```

## Getting Help

### Resources
1. **NumPy documentation**: https://numpy.org/doc/
2. **SciPy documentation**: https://docs.scipy.org/doc/scipy/
3. **spglib documentation**: https://spglib.github.io/spglib/python-spglib.html
4. **Original paper**: Computer Physics Communications, 271 (2022) 108196

### Ask Questions
- Include module name and function
- Show C code and attempted Python code
- Describe what you expected vs what happened
- Include error messages

## Progress Tracking

Update these documents as you go:
1. `README.md` - Status table
2. Module file - Remove TODO, add implementation
3. Test file - Remove placeholder, add real tests
4. Git commits - Regular commits with clear messages

Example commit messages:
- "Implement constants.py module"
- "Add tests for vector operations"
- "Fix rotation matrix calculation in rotate_orbital.py"

## Next Steps

1. **Start with Phase 1**: `constants.py`
2. **Create a PR per module** (or small group of related modules)
3. **Get review before moving to next module**
4. **Keep C code as reference**
5. **Document any deviations from C code**

## Example: Translating a Simple Function

C code:
```c
double vector_norm(vector v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
```

Python code:
```python
def vector_norm(v: Vector) -> float:
    """
    Calculate the Euclidean norm of a vector.

    Parameters
    ----------
    v : Vector
        Input vector

    Returns
    -------
    float
        Euclidean norm of the vector

    Examples
    --------
    >>> v = Vector(3.0, 4.0, 0.0)
    >>> vector_norm(v)
    5.0
    """
    return np.sqrt(v.x**2 + v.y**2 + v.z**2)
```

Test:
```python
def test_vector_norm():
    # Test known case: 3-4-5 triangle
    v = Vector(3.0, 4.0, 0.0)
    assert abs(vector_norm(v) - 5.0) < 1e-10
    
    # Test zero vector
    v = Vector(0.0, 0.0, 0.0)
    assert vector_norm(v) == 0.0
    
    # Test unit vectors
    assert abs(vector_norm(Vector(1, 0, 0)) - 1.0) < 1e-10
```

Good luck with the translation! 🚀
