#include "unity/unity.h"
#include "../src/vector.h"
#include "../src/constants.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {
    // This is run before EACH test
}

void tearDown(void) {
    // This is run after EACH test
}

// Test init_vector
void test_init_vector(void) {
    vector v;
    init_vector(&v, 1.0, 2.0, 3.0);
    
    TEST_ASSERT_TRUE(fabs(1.0 - v.x) < eps6);
    TEST_ASSERT_TRUE(fabs(2.0 - v.y) < eps6);
    TEST_ASSERT_TRUE(fabs(3.0 - v.z) < eps6);
}

// Test vector_add
void test_vector_add(void) {
    vector v1, v2, result;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    
    result = vector_add(v1, v2);
    
    TEST_ASSERT_TRUE(fabs(5.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(7.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(9.0 - result.z) < eps6);
}

// Test vector_sub
void test_vector_sub(void) {
    vector v1, v2, result;
    init_vector(&v1, 5.0, 7.0, 9.0);
    init_vector(&v2, 1.0, 2.0, 3.0);
    
    result = vector_sub(v1, v2);
    
    TEST_ASSERT_TRUE(fabs(4.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(5.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(6.0 - result.z) < eps6);
}

// Test vector_scale
void test_vector_scale(void) {
    vector v, result;
    init_vector(&v, 1.0, 2.0, 3.0);
    
    result = vector_scale(2.0, v);
    
    TEST_ASSERT_TRUE(fabs(2.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(4.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(6.0 - result.z) < eps6);
}

// Test vector_scale with zero
void test_vector_scale_zero(void) {
    vector v, result;
    init_vector(&v, 1.0, 2.0, 3.0);
    
    result = vector_scale(0.0, v);
    
    TEST_ASSERT_TRUE(fabs(0.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(0.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(0.0 - result.z) < eps6);
}

// Test dot_product
void test_dot_product(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    
    double result = dot_product(v1, v2);
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    TEST_ASSERT_TRUE(fabs(32.0 - result) < eps6);
}

// Test dot_product orthogonal vectors
void test_dot_product_orthogonal(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 0.0, 0.0);
    init_vector(&v2, 0.0, 1.0, 0.0);
    
    double result = dot_product(v1, v2);
    
    TEST_ASSERT_TRUE(fabs(0.0 - result) < eps6);
}

// Test cross_product
void test_cross_product(void) {
    vector v1, v2, result;
    init_vector(&v1, 1.0, 0.0, 0.0);
    init_vector(&v2, 0.0, 1.0, 0.0);
    
    result = cross_product(v1, v2);
    
    TEST_ASSERT_TRUE(fabs(0.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(0.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(1.0 - result.z) < eps6);
}

// Test cross_product anti-commutativity
void test_cross_product_anticommutative(void) {
    vector v1, v2, r1, r2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    
    r1 = cross_product(v1, v2);
    r2 = cross_product(v2, v1);
    
    TEST_ASSERT_TRUE(fabs(-r1.x - r2.x) < eps6);
    TEST_ASSERT_TRUE(fabs(-r1.y - r2.y) < eps6);
    TEST_ASSERT_TRUE(fabs(-r1.z - r2.z) < eps6);
}

// Test vector_norm
void test_vector_norm(void) {
    vector v;
    init_vector(&v, 3.0, 4.0, 0.0);
    
    double norm = vector_norm(v);
    
    TEST_ASSERT_TRUE(fabs(5.0 - norm) < eps6);
}

// Test vector_norm unit vector
void test_vector_norm_unit(void) {
    vector v;
    init_vector(&v, 1.0, 0.0, 0.0);
    
    double norm = vector_norm(v);
    
    TEST_ASSERT_TRUE(fabs(1.0 - norm) < eps6);
}

// Test vector_normalization
void test_vector_normalization(void) {
    vector v, result;
    init_vector(&v, 3.0, 4.0, 0.0);
    
    result = vector_normalization(v);
    
    double norm = vector_norm(result);
    TEST_ASSERT_TRUE(fabs(1.0 - norm) < eps6);
}

// Test equal
void test_equal_true(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 1.0, 2.0, 3.0);
    
    TEST_ASSERT_TRUE(equal(v1, v2));
}

// Test equal false
void test_equal_false(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 1.0, 2.0, 3.1);
    
    TEST_ASSERT_FALSE(equal(v1, v2));
}

// Test distance
void test_distance(void) {
    vector v1, v2;
    init_vector(&v1, 0.0, 0.0, 0.0);
    init_vector(&v2, 3.0, 4.0, 0.0);
    
    double dist = distance(v1, v2, NULL);
    
    TEST_ASSERT_TRUE(fabs(5.0 - dist) < eps6);
}

// Test distance zero
void test_distance_zero(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 1.0, 2.0, 3.0);
    
    double dist = distance(v1, v2, NULL);
    
    TEST_ASSERT_TRUE(fabs(0.0 - dist) < eps6);
}

// Test volume_product
void test_volume_product(void) {
    vector v1, v2, v3;
    init_vector(&v1, 1.0, 0.0, 0.0);
    init_vector(&v2, 0.0, 1.0, 0.0);
    init_vector(&v3, 0.0, 0.0, 1.0);
    
    double vol = volume_product(v1, v2, v3);
    
    TEST_ASSERT_TRUE(fabs(1.0 - vol) < eps6);
}

// Test volume_product coplanar
void test_volume_product_coplanar(void) {
    vector v1, v2, v3;
    init_vector(&v1, 1.0, 0.0, 0.0);
    init_vector(&v2, 0.0, 1.0, 0.0);
    init_vector(&v3, 1.0, 1.0, 0.0);
    
    double vol = volume_product(v1, v2, v3);
    
    TEST_ASSERT_TRUE(fabs(0.0 - vol) < eps6);
}

// Test vector_round
void test_vector_round(void) {
    vector v, result;
    init_vector(&v, 1.4, 2.6, 3.5);
    
    result = vector_round(v);
    
    TEST_ASSERT_TRUE(fabs(1.0 - result.x) < eps6);
    TEST_ASSERT_TRUE(fabs(3.0 - result.y) < eps6);
    TEST_ASSERT_TRUE(fabs(4.0 - result.z) < eps6);
}

// Test array2vector
void test_array2vector(void) {
    double arr[3] = {1.5, 2.5, 3.5};
    
    vector v = array2vector(arr);
    
    TEST_ASSERT_TRUE(fabs(1.5 - v.x) < eps6);
    TEST_ASSERT_TRUE(fabs(2.5 - v.y) < eps6);
    TEST_ASSERT_TRUE(fabs(3.5 - v.z) < eps6);
}

// Test vec_comp equal vectors
void test_vec_comp_equal(void) {
    vector v1, v2;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 1.0, 2.0, 3.0);
    
    int result = vec_comp(v1, v2, eps6);
    
    TEST_ASSERT_EQUAL_INT(0, result);
}

// Test vec_llist_init
void test_vec_llist_init(void) {
    vec_llist *head = NULL;
    
    vec_llist_init(&head);
    
    TEST_ASSERT_NULL(head);
}

// Test vec_llist_add
void test_vec_llist_add(void) {
    vec_llist *head = NULL;
    vector v;
    init_vector(&v, 1.0, 2.0, 3.0);
    
    vec_llist_init(&head);
    vec_llist_add(&head, v);
    
    TEST_ASSERT_NOT_NULL(head);
    TEST_ASSERT_TRUE(equal(head->val, v));
    TEST_ASSERT_NULL(head->next);
    
    vec_llist_free(&head);
}

// Test vec_llist_add multiple elements
void test_vec_llist_add_multiple(void) {
    vec_llist *head = NULL;
    vector v1, v2, v3;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    init_vector(&v3, 7.0, 8.0, 9.0);
    
    vec_llist_init(&head);
    vec_llist_add(&head, v1);
    vec_llist_add(&head, v2);
    vec_llist_add(&head, v3);
    
    TEST_ASSERT_NOT_NULL(head);
    TEST_ASSERT_TRUE(equal(head->val, v1));
    TEST_ASSERT_NOT_NULL(head->next);
    TEST_ASSERT_TRUE(equal(head->next->val, v2));
    TEST_ASSERT_NOT_NULL(head->next->next);
    TEST_ASSERT_TRUE(equal(head->next->next->val, v3));
    
    vec_llist_free(&head);
}

// Test vec_llist_pop
void test_vec_llist_pop(void) {
    vec_llist *head = NULL;
    vector v1, v2, popped;
    int err;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    
    vec_llist_init(&head);
    vec_llist_add(&head, v1);
    vec_llist_add(&head, v2);
    
    popped = vec_llist_pop(&head, &err);
    
    TEST_ASSERT_EQUAL_INT(1, err);
    TEST_ASSERT_TRUE(equal(popped, v1));
    TEST_ASSERT_NOT_NULL(head);
    TEST_ASSERT_TRUE(equal(head->val, v2));
    
    vec_llist_free(&head);
}

// Test vec_llist_pop empty list
void test_vec_llist_pop_empty(void) {
    vec_llist *head = NULL;
    vector popped;
    int err;
    
    vec_llist_init(&head);
    popped = vec_llist_pop(&head, &err);
    
    TEST_ASSERT_EQUAL_INT(-1, err);
}

// Test vec_llist_find
void test_vec_llist_find_existing(void) {
    vec_llist *head = NULL;
    vector v1, v2, v3;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    init_vector(&v3, 7.0, 8.0, 9.0);
    
    vec_llist_init(&head);
    vec_llist_add(&head, v1);
    vec_llist_add(&head, v2);
    vec_llist_add(&head, v3);
    
    int idx = vec_llist_find(&head, v2);
    
    TEST_ASSERT_EQUAL_INT(1, idx);
    
    vec_llist_free(&head);
}

// Test vec_llist_find not existing
void test_vec_llist_find_not_existing(void) {
    vec_llist *head = NULL;
    vector v1, v2, v_search;
    init_vector(&v1, 1.0, 2.0, 3.0);
    init_vector(&v2, 4.0, 5.0, 6.0);
    init_vector(&v_search, 10.0, 11.0, 12.0);
    
    vec_llist_init(&head);
    vec_llist_add(&head, v1);
    vec_llist_add(&head, v2);
    
    int idx = vec_llist_find(&head, v_search);
    
    TEST_ASSERT_EQUAL_INT(-1, idx);
    
    vec_llist_free(&head);
}

int main(void) {
    UNITY_BEGIN();
    
    // Basic vector operations
    RUN_TEST(test_init_vector);
    RUN_TEST(test_vector_add);
    RUN_TEST(test_vector_sub);
    RUN_TEST(test_vector_scale);
    RUN_TEST(test_vector_scale_zero);
    
    // Vector products
    RUN_TEST(test_dot_product);
    RUN_TEST(test_dot_product_orthogonal);
    RUN_TEST(test_cross_product);
    RUN_TEST(test_cross_product_anticommutative);
    RUN_TEST(test_volume_product);
    RUN_TEST(test_volume_product_coplanar);
    
    // Norms and distances
    RUN_TEST(test_vector_norm);
    RUN_TEST(test_vector_norm_unit);
    RUN_TEST(test_vector_normalization);
    RUN_TEST(test_distance);
    RUN_TEST(test_distance_zero);
    
    // Comparison and rounding
    RUN_TEST(test_equal_true);
    RUN_TEST(test_equal_false);
    RUN_TEST(test_vector_round);
    RUN_TEST(test_vec_comp_equal);
    
    // Utility functions
    RUN_TEST(test_array2vector);
    
    // Linked list operations
    RUN_TEST(test_vec_llist_init);
    RUN_TEST(test_vec_llist_add);
    RUN_TEST(test_vec_llist_add_multiple);
    RUN_TEST(test_vec_llist_pop);
    RUN_TEST(test_vec_llist_pop_empty);
    RUN_TEST(test_vec_llist_find_existing);
    RUN_TEST(test_vec_llist_find_not_existing);
    
    return UNITY_END();
}
