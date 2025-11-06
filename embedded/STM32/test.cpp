#include <stdint.h>

uint32_t test = 0;

typedef enum {
    TEST_OK = 0,
    TEST_ERROR
} TestStatus_t;

TestStatus_t testFunction(void) {
    return TEST_OK;
}
