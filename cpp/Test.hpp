#ifndef __TEST_HPP__
#define __TEST_HPP__

#include "math.h"
#include <iostream>

#define WHITE "\033[0m"
#define RED "\033[91m"
#define GREEN "\033[92m"
#define PINK "\033[95m"
#define YELLOW "\033[93m"

#define assert(expr)                                                                               \
    {                                                                                              \
        if (!(expr)) {                                                                             \
            fprintf(                                                                               \
              stderr,                                                                              \
              "\t\t%sFailed%s.\nAssertion '%s' failed line %s%d%s (function %s) in file %s.\n",    \
              RED,                                                                                 \
              WHITE,                                                                               \
              #expr,                                                                               \
              PINK,                                                                                \
              __LINE__,                                                                            \
              WHITE,                                                                               \
              __func__,                                                                            \
              __FILE__);                                                                           \
            FAILS++;                                                                               \
        }                                                                                          \
    }

#define start                                                                                      \
    printf("Starting test %s%-30s%s", YELLOW, __func__, WHITE);                                                      \
    fflush(stdout);                                                                                \
    int FAILS = 0;

#define end                                                                                        \
    if (FAILS == 0)                                                                                \
    printf("\t\t%sPassed%s\n", GREEN, WHITE)

#define close(x, y) (fabs((x) - (y)) <= (1e-5))

#endif // __TEST_HPP__