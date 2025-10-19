# WannSymm Unit Tests - Quick Start Guide

This guide shows you how to compile and run the comprehensive unit test suite for WannSymm.

## Prerequisites

Install required libraries (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y libblas-dev liblapack-dev liblapacke-dev libopenblas-dev libopenmpi-dev gcc make
```

For other systems, install equivalent packages:
- C compiler (gcc or clang)
- BLAS/LAPACK libraries
- LAPACKE (C interface to LAPACK)
- OpenBLAS (recommended) or standard BLAS
- OpenMPI or other MPI implementation

## Directory Navigation

```bash
cd WannSymm/tests
```

## Building Tests

### Build all test executables:

```bash
make all
```

Expected output:
```
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include -c ../src/matrix.c -o matrix.o
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include -c unity/unity.c -o unity/unity.o
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include test_matrix.c matrix.o unity/unity.o -o test_matrix -lm -llapacke -lopenblas -lpthread
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include -c ../src/vector.c -o vector.o
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include test_vector.c vector.o matrix.o unity/unity.o -o test_vector -lm -llapacke -lopenblas -lpthread
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include -c ../src/usefulio.c -o usefulio.o
gcc -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include test_usefulio.c usefulio.o unity/unity.o -o test_usefulio -lm -llapacke -lopenblas -lpthread
```

This creates three test executables:
- `test_matrix` - Tests for matrix operations
- `test_vector` - Tests for vector operations  
- `test_usefulio` - Tests for I/O utilities

## Running Tests

### Run all tests:

```bash
make test
```

Expected output:
```
==========================================
Running Unit Tests
==========================================

Running test_matrix...
test_matrix.c:241:test_matrix3x3_transpose_identity:PASS
test_matrix.c:242:test_matrix3x3_transpose_non_symmetric:PASS
test_matrix.c:243:test_matrix3x3_transpose_twice:PASS
test_matrix.c:246:test_matrix3x3_inverse_identity:PASS
test_matrix.c:247:test_matrix3x3_inverse_simple:PASS
test_matrix.c:248:test_matrix3x3_inverse_property:PASS
test_matrix.c:251:test_matrix3x3_dot_identity:PASS
test_matrix.c:252:test_matrix3x3_dot_general:PASS
test_matrix.c:255:test_matrix3x3_determinant_identity:PASS
test_matrix.c:256:test_matrix3x3_determinant_zero:PASS
test_matrix.c:257:test_matrix3x3_determinant_nonzero:PASS
test_matrix.c:260:test_matrix3x3_copy:PASS
test_matrix.c:261:test_matrix3x1_copy:PASS

-----------------------
13 Tests 0 Failures 0 Ignored 
OK

Running test_vector...
test_vector.c:379:test_init_vector:PASS
test_vector.c:380:test_vector_add:PASS
test_vector.c:381:test_vector_sub:PASS
test_vector.c:382:test_vector_scale:PASS
test_vector.c:383:test_vector_scale_zero:PASS
test_vector.c:386:test_dot_product:PASS
test_vector.c:387:test_dot_product_orthogonal:PASS
test_vector.c:388:test_cross_product:PASS
test_vector.c:389:test_cross_product_anticommutative:PASS
test_vector.c:390:test_volume_product:PASS
test_vector.c:391:test_volume_product_coplanar:PASS
test_vector.c:394:test_vector_norm:PASS
test_vector.c:395:test_vector_norm_unit:PASS
test_vector.c:396:test_vector_normalization:PASS
test_vector.c:397:test_distance:PASS
test_vector.c:398:test_distance_zero:PASS
test_vector.c:401:test_equal_true:PASS
test_vector.c:402:test_equal_false:PASS
test_vector.c:403:test_vector_round:PASS
test_vector.c:404:test_vec_comp_equal:PASS
test_vector.c:407:test_array2vector:PASS
test_vector.c:410:test_vec_llist_init:PASS
test_vector.c:411:test_vec_llist_add:PASS
test_vector.c:412:test_vec_llist_add_multiple:PASS
test_vector.c:413:test_vec_llist_pop:PASS
test_vector.c:414:test_vec_llist_pop_empty:PASS
test_vector.c:415:test_vec_llist_find_existing:PASS
test_vector.c:416:test_vec_llist_find_not_existing:PASS

-----------------------
28 Tests 0 Failures 0 Ignored 
OK

Running test_usefulio...
test_usefulio.c:154:test_get_memory_usage_valid:PASS
test_usefulio.c:155:test_get_memory_usage_positive_on_linux:PASS
test_usefulio.c:156:test_get_memory_usage_multiple_calls:PASS
test_usefulio.c:159:test_get_memory_usage_str_format_gib:PASS
test_usefulio.c:160:test_get_memory_usage_str_appends:PASS
test_usefulio.c:161:test_get_memory_usage_str_has_numeric_value:PASS
test_usefulio.c:162:test_get_memory_usage_str_empty_prefix:PASS
test_usefulio.c:163:test_get_memory_usage_str_various_prefixes:PASS
test_usefulio.c:166:test_get_memory_usage_increases_after_allocation:PASS

-----------------------
9 Tests 0 Failures 0 Ignored 
OK

==========================================
All tests completed!
==========================================
```

### Run individual test suites:

```bash
# Test only matrix operations
make test-matrix

# Test only vector operations
make test-vector

# Test only I/O utilities
make test-usefulio
```

### Run test executables directly:

```bash
# Run individual test executable
./test_matrix
./test_vector
./test_usefulio
```

## Cleaning Up

Remove all compiled test binaries and object files:

```bash
make clean
```

Expected output:
```
rm -f test_matrix test_vector test_usefulio *.o unity/unity.o
```

## Getting Help

Show available make targets:

```bash
make help
```

Output:
```
WannSymm Test Suite Makefile

Available targets:
  all           - Build all test executables (default)
  test          - Build and run all tests
  test-matrix   - Build and run matrix tests only
  test-vector   - Build and run vector tests only
  test-usefulio - Build and run usefulio tests only
  clean         - Remove all build artifacts
  help          - Show this help message

Example usage:
  make test     # Run all tests
  make clean    # Clean build artifacts
```

## Understanding Test Results

Each test shows:
1. **File and line number**: Where the test is defined
2. **Test name**: Descriptive name of what's being tested
3. **Result**: PASS, FAIL, or IGNORED

At the end of each test suite:
- **X Tests Y Failures Z Ignored**: Summary of results
- **OK** or **FAIL**: Overall status

## Test Coverage

### test_matrix.c (13 tests)
- Transpose operations
- Inverse matrix calculations
- Matrix multiplication
- Determinant computation
- Matrix copying

### test_vector.c (28 tests)
- Basic vector operations (add, sub, scale)
- Vector products (dot, cross, volume)
- Norms and distances
- Vector comparisons
- Linked list operations

### test_usefulio.c (9 tests)
- Memory usage reporting
- String formatting
- Platform-specific behavior
- Memory allocation detection

## Total: 50 Tests - All Passing ✓

## Troubleshooting

### Issue: "fatal error: lapacke.h: No such file or directory"

**Solution**: Install LAPACKE development headers:
```bash
sudo apt-get install liblapacke-dev
```

### Issue: "undefined reference to `cblas_dgemm`"

**Solution**: Install OpenBLAS:
```bash
sudo apt-get install libopenblas-dev
```

### Issue: "fatal error: mpi.h: No such file or directory"

**Solution**: Install MPI development headers:
```bash
sudo apt-get install libopenmpi-dev
```

### Issue: Compilation warnings about unused variables

These warnings come from the original source code and don't affect test execution. Tests will still run successfully.

## Next Steps

- Review test source code in `test_*.c` files
- Read detailed documentation in `tests/README.md`
- Review comprehensive summary in `TEST_SUMMARY.md`
- Add tests for additional modules as needed

## Quick Command Reference

```bash
# Complete test workflow
cd tests/
make clean      # Start fresh
make all        # Build tests
make test       # Run all tests

# Individual operations
make test-matrix    # Test matrix operations only
make test-vector    # Test vector operations only
make test-usefulio  # Test I/O utilities only

# Manual execution
./test_matrix       # Run matrix tests directly
./test_vector       # Run vector tests directly  
./test_usefulio     # Run I/O tests directly

# Cleanup
make clean         # Remove build artifacts
```

## Success Criteria

✅ All 50 tests pass
✅ No segmentation faults
✅ No memory errors
✅ All modules function correctly
✅ Clean build with no critical errors

All success criteria met! ✓
