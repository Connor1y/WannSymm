# WannSymm Unit Test Implementation Summary

## Overview

This document summarizes the comprehensive unit test framework implemented for the WannSymm project. The tests cover core utility modules in the src/ directory using the Unity testing framework.

## Directory Structure

```
WannSymm/
├── src/                          # Source files
│   ├── matrix.c                  # Matrix operations (tested)
│   ├── vector.c                  # Vector operations (tested)
│   ├── usefulio.c               # I/O utilities (tested)
│   └── ... (other source files)
└── tests/                        # Test suite (NEW)
    ├── unity/                    # Unity testing framework
    │   ├── unity.c
    │   ├── unity.h
    │   └── unity_internals.h
    ├── mkl.h                     # MKL compatibility header
    ├── spglib.h                  # spglib stub header
    ├── test_matrix.c             # Matrix operation tests
    ├── test_vector.c             # Vector operation tests
    ├── test_usefulio.c           # I/O utility tests
    ├── Makefile                  # Build system for tests
    └── README.md                 # Test documentation
```

## Test Framework: Unity

Unity is a lightweight C testing framework specifically designed for embedded systems and resource-constrained environments. It provides:
- Simple, readable test syntax
- No dynamic memory allocation
- Platform independence
- Minimal dependencies

Source: https://github.com/ThrowTheSwitch/Unity

## Test Coverage

### 1. test_matrix.c - 13 Tests

Tests for `src/matrix.c` covering:

**matrix3x3_transpose** (3 tests):
- Identity matrix transpose
- Non-symmetric matrix transpose
- Double transpose property (A^T^T = A)

**matrix3x3_inverse** (3 tests):
- Identity matrix inverse
- Diagonal matrix inverse
- Inverse property (A * A^-1 = I)

**matrix3x3_dot** (2 tests):
- Identity matrix multiplication
- General matrix multiplication

**matrix3x3_determinant** (3 tests):
- Identity matrix determinant (should be 1)
- Zero determinant (singular matrix)
- Non-zero determinant

**matrix3x3_copy and matrix3x1_copy** (2 tests):
- Matrix copying
- Vector copying

### 2. test_vector.c - 28 Tests

Tests for `src/vector.c` covering:

**Basic operations** (5 tests):
- Vector initialization
- Vector addition
- Vector subtraction
- Scalar multiplication
- Scalar multiplication by zero

**Vector products** (6 tests):
- Dot product (standard case)
- Dot product (orthogonal vectors)
- Cross product (standard case)
- Cross product anti-commutativity (v1 × v2 = -(v2 × v1))
- Triple scalar product (volume)
- Triple scalar product (coplanar vectors)

**Norms and distances** (5 tests):
- Vector norm (3-4-5 triangle)
- Unit vector norm
- Vector normalization
- Distance between vectors
- Zero distance (same point)

**Comparison and utilities** (4 tests):
- Vector equality (true case)
- Vector equality (false case)
- Vector rounding
- Vector comparison function

**Conversion** (1 test):
- Array to vector conversion

**Linked list operations** (7 tests):
- List initialization
- Add single element
- Add multiple elements
- Pop element
- Pop from empty list (error case)
- Find existing element
- Find non-existing element

### 3. test_usefulio.c - 9 Tests

Tests for `src/usefulio.c` covering:

**get_memory_usage** (3 tests):
- Returns valid value (non-negative or error code)
- Returns positive value on supported platforms (Linux, macOS)
- Multiple consecutive calls work correctly

**get_memory_usage_str** (5 tests):
- Formats with correct units (GiB, MiB, KiB)
- Appends to existing string prefix
- Includes numeric value
- Handles empty prefix
- Works with various prefixes

**Memory allocation detection** (1 test):
- Memory usage increases after allocation (platform-specific)

## Build System

### Makefile Features

The test Makefile provides:
- Automatic dependency handling
- Support for both MKL and standard BLAS/LAPACK
- Individual test execution
- Comprehensive clean target
- Help documentation

### Key Targets

```bash
make all          # Build all test executables
make test         # Build and run all tests
make test-matrix  # Run only matrix tests
make test-vector  # Run only vector tests
make test-usefulio # Run only usefulio tests
make clean        # Remove build artifacts
make help         # Show usage information
```

## Compatibility Headers

To enable testing without full project dependencies:

### mkl.h
A compatibility header that maps to standard LAPACKE and CBLAS when Intel MKL is not available.

### spglib.h
A stub header providing minimal type definitions for spglib structures, allowing compilation without spglib installation.

## How to Build and Run Tests

### Prerequisites
```bash
# Install required libraries (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libblas-dev liblapack-dev liblapacke-dev libopenblas-dev libopenmpi-dev
```

### Building Tests
```bash
cd tests
make all
```

### Running All Tests
```bash
make test
```

### Expected Output
```
==========================================
Running Unit Tests
==========================================

Running test_matrix...
-----------------------
13 Tests 0 Failures 0 Ignored 
OK

Running test_vector...
-----------------------
28 Tests 0 Failures 0 Ignored 
OK

Running test_usefulio...
-----------------------
9 Tests 0 Failures 0 Ignored 
OK

==========================================
All tests completed!
==========================================
```

## Test Statistics

- **Total Tests**: 50
- **Total Test Files**: 3
- **Code Coverage**: Core utility modules (matrix, vector, usefulio)
- **Test Success Rate**: 100% (50/50 passing)

## Example Test Case

Here's an example test from test_matrix.c:

```c
// Test matrix3x3_inverse with identity matrix
void test_matrix3x3_inverse_identity(void) {
    double in[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    double out[3][3];
    double expected[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    matrix3x3_inverse(out, in);
    TEST_ASSERT_TRUE(matrices_equal(out, expected, TOLERANCE));
}
```

## Adding New Tests

To add tests for additional modules:

1. Create `tests/test_<module>.c`
2. Include Unity and the module header:
   ```c
   #include "unity/unity.h"
   #include "../src/<module>.h"
   ```
3. Implement setUp/tearDown functions
4. Write test functions (prefix with `test_`)
5. Add to Makefile:
   - Add object file to variables
   - Add build rule
   - Add to TESTS variable
   - Add individual test target

## Testing Philosophy

The tests follow these principles:

1. **Comprehensive Coverage**: Test normal cases, edge cases, and error conditions
2. **Independence**: Each test is independent and can run alone
3. **Readability**: Test names clearly describe what is being tested
4. **Minimal**: No unnecessary complexity or dependencies
5. **Maintainability**: Easy to add new tests or modify existing ones

## Platform Support

The tests compile and run on:
- Linux (tested on Ubuntu)
- macOS (with appropriate libraries)
- Any platform with:
  - C compiler (gcc, clang)
  - BLAS/LAPACK or OpenBLAS
  - LAPACKE
  - MPI (for vector.c dependencies)

## Future Enhancements

Potential additions to the test suite:

1. Tests for larger modules (readinput.c, rotate_ham.c, etc.)
2. Integration tests for module interactions
3. Performance benchmarks
4. Code coverage analysis with gcov/lcov
5. Continuous integration setup
6. Fuzzing for robustness testing
7. Memory leak detection with valgrind

## Conclusion

This comprehensive test suite provides:
- ✅ 50 unit tests covering 3 core modules
- ✅ Clean, maintainable test infrastructure
- ✅ Easy-to-use build system
- ✅ Platform-independent compatibility
- ✅ Clear documentation for developers
- ✅ Foundation for expanded test coverage

All tests pass successfully, demonstrating that the tested modules work correctly for their intended use cases.
