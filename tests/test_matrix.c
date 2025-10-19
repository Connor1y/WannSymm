#include "unity/unity.h"
#include "../src/matrix.h"
#include <math.h>
#include <string.h>

// Tolerance for floating point comparisons
#define TOLERANCE 1e-10

void setUp(void) {
    // This is run before EACH test
}

void tearDown(void) {
    // This is run after EACH test
}

// Helper function to check if two 3x3 matrices are equal
int matrices_equal(double a[3][3], double b[3][3], double tol) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(a[i][j] - b[i][j]) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

// Test matrix3x3_transpose with identity matrix
void test_matrix3x3_transpose_identity(void) {
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
    
    matrix3x3_transpose(out, in);
    TEST_ASSERT_TRUE(matrices_equal(out, expected, TOLERANCE));
}

// Test matrix3x3_transpose with non-symmetric matrix
void test_matrix3x3_transpose_non_symmetric(void) {
    double in[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    double out[3][3];
    double expected[3][3] = {
        {1.0, 4.0, 7.0},
        {2.0, 5.0, 8.0},
        {3.0, 6.0, 9.0}
    };
    
    matrix3x3_transpose(out, in);
    TEST_ASSERT_TRUE(matrices_equal(out, expected, TOLERANCE));
}

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

// Test matrix3x3_inverse with simple matrix
void test_matrix3x3_inverse_simple(void) {
    double in[3][3] = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0}
    };
    double out[3][3];
    double expected[3][3] = {
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5}
    };
    
    matrix3x3_inverse(out, in);
    TEST_ASSERT_TRUE(matrices_equal(out, expected, TOLERANCE));
}

// Test matrix3x3_dot with identity matrix
void test_matrix3x3_dot_identity(void) {
    double a[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    double b[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    double out[3][3];
    
    matrix3x3_dot(out, a, b);
    TEST_ASSERT_TRUE(matrices_equal(out, a, TOLERANCE));
}

// Test matrix3x3_dot with general matrices
void test_matrix3x3_dot_general(void) {
    double a[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    double b[3][3] = {
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}
    };
    double out[3][3];
    double expected[3][3] = {
        {30.0, 24.0, 18.0},
        {84.0, 69.0, 54.0},
        {138.0, 114.0, 90.0}
    };
    
    matrix3x3_dot(out, a, b);
    TEST_ASSERT_TRUE(matrices_equal(out, expected, TOLERANCE));
}

// Test matrix3x3_determinant with identity matrix
void test_matrix3x3_determinant_identity(void) {
    double m[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    double det = matrix3x3_determinant(m);
    TEST_ASSERT_TRUE(fabs(det - 1.0) < TOLERANCE);
}

// Test matrix3x3_determinant with zero determinant
void test_matrix3x3_determinant_zero(void) {
    double m[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    double det = matrix3x3_determinant(m);
    TEST_ASSERT_TRUE(fabs(det - 0.0) < TOLERANCE);
}

// Test matrix3x3_determinant with non-zero determinant
void test_matrix3x3_determinant_nonzero(void) {
    double m[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };
    
    double det = matrix3x3_determinant(m);
    TEST_ASSERT_TRUE(fabs(det - 6.0) < TOLERANCE);
}

// Test matrix3x3_copy
void test_matrix3x3_copy(void) {
    double in[3][3] = {
        {1.5, 2.5, 3.5},
        {4.5, 5.5, 6.5},
        {7.5, 8.5, 9.5}
    };
    double out[3][3];
    
    matrix3x3_copy(out, in);
    TEST_ASSERT_TRUE(matrices_equal(out, in, TOLERANCE));
}

// Test matrix3x1_copy
void test_matrix3x1_copy(void) {
    double in[3] = {1.5, 2.5, 3.5};
    double out[3];
    
    matrix3x1_copy(out, in);
    TEST_ASSERT_TRUE(fabs(in[0] - out[0]) < TOLERANCE);
    TEST_ASSERT_TRUE(fabs(in[1] - out[1]) < TOLERANCE);
    TEST_ASSERT_TRUE(fabs(in[2] - out[2]) < TOLERANCE);
}

// Test double transpose returns original
void test_matrix3x3_transpose_twice(void) {
    double in[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    double tmp[3][3];
    double out[3][3];
    
    matrix3x3_transpose(tmp, in);
    matrix3x3_transpose(out, tmp);
    TEST_ASSERT_TRUE(matrices_equal(out, in, TOLERANCE));
}

// Test inverse times original equals identity
void test_matrix3x3_inverse_property(void) {
    double in[3][3] = {
        {1.0, 2.0, 0.0},
        {0.0, 3.0, 1.0},
        {1.0, 0.0, 2.0}
    };
    double inv[3][3];
    double result[3][3];
    double identity[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    matrix3x3_inverse(inv, in);
    matrix3x3_dot(result, in, inv);
    TEST_ASSERT_TRUE(matrices_equal(result, identity, 1e-6));
}

int main(void) {
    UNITY_BEGIN();
    
    // Transpose tests
    RUN_TEST(test_matrix3x3_transpose_identity);
    RUN_TEST(test_matrix3x3_transpose_non_symmetric);
    RUN_TEST(test_matrix3x3_transpose_twice);
    
    // Inverse tests
    RUN_TEST(test_matrix3x3_inverse_identity);
    RUN_TEST(test_matrix3x3_inverse_simple);
    RUN_TEST(test_matrix3x3_inverse_property);
    
    // Dot product tests
    RUN_TEST(test_matrix3x3_dot_identity);
    RUN_TEST(test_matrix3x3_dot_general);
    
    // Determinant tests
    RUN_TEST(test_matrix3x3_determinant_identity);
    RUN_TEST(test_matrix3x3_determinant_zero);
    RUN_TEST(test_matrix3x3_determinant_nonzero);
    
    // Copy tests
    RUN_TEST(test_matrix3x3_copy);
    RUN_TEST(test_matrix3x1_copy);
    
    return UNITY_END();
}
