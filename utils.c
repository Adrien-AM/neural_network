#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float relu(float x, int derivative)
{
    if (derivative)
        return x >= 0 ? 1 : 0;
    return x >= 0 ? x : 0;
}

float sigmoid(float x, int derivative)
{
    if (derivative)
    {
        float out = sigmoid(x, 0);
        return out * (1 - out);
    }

    return 1 / (1 + exp(-x));
}

float linear(float x, int derivative)
{
    if(derivative)
        return 1;
    return x;
}

// CREDITS https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus

float rand_gen()
{
    return ((float)rand() + 1) / ((float)RAND_MAX + 1);
}

float rand_normal(float mu, float sigma)
{
    float v1 = rand_gen();
    float v2 = rand_gen();

    float r1 = cos(2 * 3.14 * v2) * sqrt(-2 * log(v1));
    return r1 * sigma + mu;
}