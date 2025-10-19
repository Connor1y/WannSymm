# WannSymm Test Suite

This directory contains comprehensive unit tests for the WannSymm source code modules.

## Test Framework

The tests use the [Unity](https://github.com/ThrowTheSwitch/Unity) testing framework, which is a lightweight C testing framework specifically designed for embedded systems and resource-constrained environments.

## Directory Structure

```
tests/
├── unity/                  # Unity testing framework
│   ├── unity.c
│   ├── unity.h
│   └── unity_internals.h
├── test_matrix.c          # Tests for matrix operations (matrix.c)
├── test_vector.c          # Tests for vector operations (vector.c)
├── test_usefulio.c        # Tests for I/O utilities (usefulio.c)
├── Makefile               # Build system for tests
└── README.md              # This file
```

## Building and Running Tests

### Prerequisites

Before running the tests, make sure you have:

1. A C compiler (gcc or compatible)
2. Standard math library (libm)
3. BLAS/LAPACK libraries OR Intel MKL

If you have MKL configured in `../make.sys`, the tests will use it automatically. Otherwise, they will fall back to standard BLAS/LAPACK.

### Build All Tests

```bash
cd tests
make all
```

### Run All Tests

```bash
make test
```

This will:
1. Build all test executables
2. Run each test suite
3. Display results for each test

### Run Individual Test Suites

```bash
# Run only matrix tests
make test-matrix

# Run only vector tests
make test-vector

# Run only usefulio tests
make test-usefulio
```

### Clean Build Artifacts

```bash
make clean
```

### Get Help

```bash
make help
```

## Test Coverage

### test_matrix.c

Tests for `src/matrix.c` covering:
- **matrix3x3_transpose**: Identity matrix, non-symmetric matrices, double transpose property
- **matrix3x3_inverse**: Identity matrix, diagonal matrices, inverse property (A * A^-1 = I)
- **matrix3x3_dot**: Identity matrix multiplication, general matrix multiplication
- **matrix3x3_determinant**: Identity, zero determinant, non-zero determinant
- **matrix3x3_copy**: Matrix copying
- **matrix3x1_copy**: Vector copying

### test_vector.c

Tests for `src/vector.c` covering:
- **init_vector**: Vector initialization
- **vector_add/sub**: Vector addition and subtraction
- **vector_scale**: Scalar multiplication, including zero scaling
- **dot_product**: Standard and orthogonal vectors
- **cross_product**: Standard cases and anti-commutativity property
- **vector_norm**: Norm calculation for various vectors
- **vector_normalization**: Normalization to unit vectors
- **equal**: Vector equality comparison
- **distance**: Distance between vectors
- **volume_product**: Triple scalar product, coplanar vectors
- **vector_round**: Rounding operations
- **array2vector**: Array to vector conversion
- **vec_comp**: Vector comparison
- **Linked list operations**: init, add, pop, find, free

### test_usefulio.c

Tests for `src/usefulio.c` covering:
- **get_memory_usage**: Memory usage retrieval on various platforms
- **get_memory_usage_str**: String formatting with various units (GiB, MiB, KiB)
- Platform-specific behavior (Linux, macOS, unsupported platforms)
- Memory allocation detection
- Multiple consecutive calls
- String prefix handling

## Expected Output

When all tests pass, you should see output similar to:

```
==========================================
Running Unit Tests
==========================================

Running test_matrix...
test_matrix.c:221:test_matrix3x3_transpose_identity:PASS
test_matrix.c:222:test_matrix3x3_transpose_non_symmetric:PASS
...
-----------------------
15 Tests 0 Failures 0 Ignored 
OK

Running test_vector...
test_vector.c:428:test_init_vector:PASS
test_vector.c:429:test_vector_add:PASS
...
-----------------------
33 Tests 0 Failures 0 Ignored 
OK

Running test_usefulio...
test_usefulio.c:137:test_get_memory_usage_valid:PASS
test_usefulio.c:138:test_get_memory_usage_positive_on_linux:PASS
...
-----------------------
10 Tests 0 Failures 0 Ignored 
OK

==========================================
All tests completed!
==========================================
```

## Adding New Tests

To add tests for a new module:

1. Create a new test file: `test_<module>.c`
2. Include Unity and the module header:
   ```c
   #include "unity/unity.h"
   #include "../src/<module>.h"
   ```
3. Implement setUp/tearDown functions
4. Write test functions with names starting with `test_`
5. Create a main function that calls `UNITY_BEGIN()`, `RUN_TEST()` for each test, and `UNITY_END()`
6. Add build rules to the Makefile
7. Add the new test to the `TESTS` variable in the Makefile

## Troubleshooting

### Missing MKL or BLAS/LAPACK

If you get linking errors about missing BLAS/LAPACK functions:

1. Install BLAS/LAPACK: `sudo apt-get install libblas-dev liblapack-dev` (Ubuntu/Debian)
2. Or configure MKL in `../make.sys` if you have Intel MKL installed

### Compiler Warnings

The tests are compiled with `-Wall` to catch potential issues. If you see warnings, they should be investigated and fixed.

### Test Failures

If tests fail:
1. Review the failure message which includes the test name and assertion that failed
2. Check if the source code behavior has changed
3. Update tests if the new behavior is correct
4. Fix bugs in the source code if the old behavior was correct

## Contributing

When contributing code to WannSymm:
1. Write tests for new functionality
2. Ensure all existing tests still pass
3. Aim for comprehensive coverage of normal cases, edge cases, and error conditions
4. Document any platform-specific behavior

## License

These tests are part of the WannSymm project and are subject to the same license terms as the main project.
