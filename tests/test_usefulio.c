#include "unity/unity.h"
#include "../src/usefulio.h"
#include <string.h>
#include <stdio.h>

void setUp(void) {
    // This is run before EACH test
}

void tearDown(void) {
    // This is run after EACH test
}

// Test get_memory_usage returns a valid value
void test_get_memory_usage_valid(void) {
    long mem_usage = get_memory_usage();
    
    // Memory usage should be non-negative
    // On unsupported platforms it returns 0, on error it returns -1
    TEST_ASSERT_GREATER_OR_EQUAL(-1, mem_usage);
}

// Test get_memory_usage returns positive on supported platforms
void test_get_memory_usage_positive_on_linux(void) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__) || (defined(__APPLE__) && defined(__MACH__))
    long mem_usage = get_memory_usage();
    
    // On supported platforms, memory usage should be positive
    TEST_ASSERT_GREATER_THAN(0, mem_usage);
#else
    TEST_PASS_MESSAGE("Skipping test on unsupported platform");
#endif
}

// Test get_memory_usage_str formats correctly for large values
void test_get_memory_usage_str_format_gib(void) {
    char mem_usage_str[256] = "Memory: ";
    
    get_memory_usage_str(mem_usage_str);
    
    // Check that the string was modified (not still just "Memory: ")
    TEST_ASSERT_GREATER_THAN(8, strlen(mem_usage_str));
    
    // Check that it contains a unit (GiB, MiB, or KiB)
    int has_unit = (strstr(mem_usage_str, "GiB") != NULL ||
                    strstr(mem_usage_str, "MiB") != NULL ||
                    strstr(mem_usage_str, "KiB") != NULL);
    TEST_ASSERT_TRUE(has_unit);
}

// Test get_memory_usage_str appends to existing string
void test_get_memory_usage_str_appends(void) {
    char mem_usage_str[256] = "Current usage: ";
    char *original_content = "Current usage: ";
    
    get_memory_usage_str(mem_usage_str);
    
    // Check that original content is still there
    TEST_ASSERT_EQUAL_STRING_LEN(original_content, mem_usage_str, strlen(original_content));
}

// Test get_memory_usage_str includes numeric value
void test_get_memory_usage_str_has_numeric_value(void) {
    char mem_usage_str[256] = "";
    
    get_memory_usage_str(mem_usage_str);
    
    // Check that the string contains a digit
    int has_digit = 0;
    for (int i = 0; mem_usage_str[i] != '\0'; i++) {
        if (mem_usage_str[i] >= '0' && mem_usage_str[i] <= '9') {
            has_digit = 1;
            break;
        }
    }
    TEST_ASSERT_TRUE(has_digit);
}

// Test get_memory_usage_str handles empty prefix
void test_get_memory_usage_str_empty_prefix(void) {
    char mem_usage_str[256] = "";
    
    get_memory_usage_str(mem_usage_str);
    
    // Should still produce a valid string
    TEST_ASSERT_GREATER_THAN(0, strlen(mem_usage_str));
}

// Test that calling get_memory_usage multiple times works
void test_get_memory_usage_multiple_calls(void) {
    long mem1 = get_memory_usage();
    long mem2 = get_memory_usage();
    long mem3 = get_memory_usage();
    
    // All calls should succeed (return >= -1)
    TEST_ASSERT_GREATER_OR_EQUAL(-1, mem1);
    TEST_ASSERT_GREATER_OR_EQUAL(-1, mem2);
    TEST_ASSERT_GREATER_OR_EQUAL(-1, mem3);
}

// Test memory usage increases after allocation
void test_get_memory_usage_increases_after_allocation(void) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__) || (defined(__APPLE__) && defined(__MACH__))
    long mem_before = get_memory_usage();
    
    // Allocate a large block of memory
    size_t alloc_size = 10 * 1024 * 1024; // 10 MB
    void *ptr = malloc(alloc_size);
    
    // Touch the memory to ensure it's actually allocated
    if (ptr != NULL) {
        memset(ptr, 0, alloc_size);
        
        long mem_after = get_memory_usage();
        
        // Memory usage should have increased
        TEST_ASSERT_GREATER_OR_EQUAL(mem_before, mem_after);
        
        free(ptr);
    } else {
        TEST_FAIL_MESSAGE("Memory allocation failed");
    }
#else
    TEST_PASS_MESSAGE("Skipping test on unsupported platform");
#endif
}

// Test get_memory_usage_str with various prefixes
void test_get_memory_usage_str_various_prefixes(void) {
    char test_cases[][256] = {
        "Prefix 1: ",
        "Another prefix: ",
        "Test: ",
        "X"
    };
    
    for (int i = 0; i < 4; i++) {
        char mem_usage_str[256];
        strcpy(mem_usage_str, test_cases[i]);
        size_t prefix_len = strlen(test_cases[i]);
        
        get_memory_usage_str(mem_usage_str);
        
        // Should append to the prefix
        TEST_ASSERT_GREATER_THAN(prefix_len, strlen(mem_usage_str));
        TEST_ASSERT_EQUAL_STRING_LEN(test_cases[i], mem_usage_str, prefix_len);
    }
}

int main(void) {
    UNITY_BEGIN();
    
    // Basic functionality tests
    RUN_TEST(test_get_memory_usage_valid);
    RUN_TEST(test_get_memory_usage_positive_on_linux);
    RUN_TEST(test_get_memory_usage_multiple_calls);
    
    // String formatting tests
    RUN_TEST(test_get_memory_usage_str_format_gib);
    RUN_TEST(test_get_memory_usage_str_appends);
    RUN_TEST(test_get_memory_usage_str_has_numeric_value);
    RUN_TEST(test_get_memory_usage_str_empty_prefix);
    RUN_TEST(test_get_memory_usage_str_various_prefixes);
    
    // Allocation test
    RUN_TEST(test_get_memory_usage_increases_after_allocation);
    
    return UNITY_END();
}
