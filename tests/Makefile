# Test Makefile for WannSymm
# This makefile builds and runs unit tests for the source modules

# Compiler settings
CC = gcc
CFLAGS = -g -Wall -I../src -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include
LDFLAGS = -lm

# MKL and SPGLIB paths (adjust if needed)
# These can be overridden by creating a make.sys in the root directory
-include ../make.sys

# If MKL is available, use it
ifdef MKLROOT
CFLAGS += -I$(MKLINCLUDE) -DMKL_ILP64 -m64
LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl
else
# Fallback to standard BLAS/LAPACK if MKL not available
# We have a local mkl.h that maps to LAPACKE and CBLAS
LDFLAGS += -llapacke -lopenblas -lpthread
endif

# Unity framework
UNITY_SRC = unity/unity.c
UNITY_OBJ = unity/unity.o

# Source files from ../src
SRC_DIR = ../src
MATRIX_SRC = $(SRC_DIR)/matrix.c
VECTOR_SRC = $(SRC_DIR)/vector.c
USEFULIO_SRC = $(SRC_DIR)/usefulio.c

# Object files
MATRIX_OBJ = matrix.o
VECTOR_OBJ = vector.o
USEFULIO_OBJ = usefulio.o

# Test executables
TEST_MATRIX = test_matrix
TEST_VECTOR = test_vector
TEST_USEFULIO = test_usefulio

# All tests
TESTS = $(TEST_MATRIX) $(TEST_VECTOR) $(TEST_USEFULIO)

# Default target
.PHONY: all
all: $(TESTS)

# Unity framework object
$(UNITY_OBJ): $(UNITY_SRC)
	$(CC) $(CFLAGS) -c $(UNITY_SRC) -o $(UNITY_OBJ)

# Compile source objects
$(MATRIX_OBJ): $(MATRIX_SRC)
	$(CC) $(CFLAGS) -c $(MATRIX_SRC) -o $(MATRIX_OBJ)

$(VECTOR_OBJ): $(VECTOR_SRC)
	$(CC) $(CFLAGS) -c $(VECTOR_SRC) -o $(VECTOR_OBJ)

$(USEFULIO_OBJ): $(USEFULIO_SRC)
	$(CC) $(CFLAGS) -c $(USEFULIO_SRC) -o $(USEFULIO_OBJ)

# Build test executables
$(TEST_MATRIX): test_matrix.c $(MATRIX_OBJ) $(UNITY_OBJ)
	$(CC) $(CFLAGS) test_matrix.c $(MATRIX_OBJ) $(UNITY_OBJ) -o $(TEST_MATRIX) $(LDFLAGS)

$(TEST_VECTOR): test_vector.c $(VECTOR_OBJ) $(MATRIX_OBJ) $(UNITY_OBJ)
	$(CC) $(CFLAGS) test_vector.c $(VECTOR_OBJ) $(MATRIX_OBJ) $(UNITY_OBJ) -o $(TEST_VECTOR) $(LDFLAGS)

$(TEST_USEFULIO): test_usefulio.c $(USEFULIO_OBJ) $(UNITY_OBJ)
	$(CC) $(CFLAGS) test_usefulio.c $(USEFULIO_OBJ) $(UNITY_OBJ) -o $(TEST_USEFULIO) $(LDFLAGS)

# Run tests
.PHONY: test
test: $(TESTS)
	@echo "=========================================="
	@echo "Running Unit Tests"
	@echo "=========================================="
	@echo ""
	@echo "Running test_matrix..."
	@./$(TEST_MATRIX)
	@echo ""
	@echo "Running test_vector..."
	@./$(TEST_VECTOR)
	@echo ""
	@echo "Running test_usefulio..."
	@./$(TEST_USEFULIO)
	@echo ""
	@echo "=========================================="
	@echo "All tests completed!"
	@echo "=========================================="

# Run individual tests
.PHONY: test-matrix
test-matrix: $(TEST_MATRIX)
	@echo "Running test_matrix..."
	@./$(TEST_MATRIX)

.PHONY: test-vector
test-vector: $(TEST_VECTOR)
	@echo "Running test_vector..."
	@./$(TEST_VECTOR)

.PHONY: test-usefulio
test-usefulio: $(TEST_USEFULIO)
	@echo "Running test_usefulio..."
	@./$(TEST_USEFULIO)

# Clean
.PHONY: clean
clean:
	rm -f $(TESTS) *.o $(UNITY_OBJ)

# Help
.PHONY: help
help:
	@echo "WannSymm Test Suite Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build all test executables (default)"
	@echo "  test          - Build and run all tests"
	@echo "  test-matrix   - Build and run matrix tests only"
	@echo "  test-vector   - Build and run vector tests only"
	@echo "  test-usefulio - Build and run usefulio tests only"
	@echo "  clean         - Remove all build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make test     # Run all tests"
	@echo "  make clean    # Clean build artifacts"
