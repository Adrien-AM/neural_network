#include "utils.h"

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
    if (derivative)
        return 1;
    return x;
}

float hypertan(float x, int derivative)
{
    if (derivative)
    {
        float r = tanh(x);
        return 1 - (r * r);
    }
    return tanh(x);
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

float mean_squared_error(float *y_true, float *y_pred, size_t size)
{
    if (0 == size)
        return 0;
    float total = 0;
    for (size_t i = 0; i < size; i++)
    {
        float error = y_true[i] - y_pred[i];
        total += error * error;
    }

    return total / size;
}

float mean_absolute_error(float *y_true, float *y_pred, size_t size)
{
    if (0 == size)
        return 0;
    float total = 0;
    for (size_t i = 0; i < size; i++)
    {
        float error = y_true[i] - y_pred[i];
        total += fabs(error);
    }
    return total / size;
}