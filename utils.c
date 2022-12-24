#include "utils.h"

double relu(double x, int derivative)
{
    if (derivative)
        return x >= 0 ? 1 : 0;
    return x >= 0 ? x : 0;
}

double sigmoid(double x, int derivative)
{
    if (derivative)
    {
        double out = sigmoid(x, 0);
        return out * (1 - out);
    }

    return 1 / (1 + exp(-x));
}

double linear(double x, int derivative)
{
    if (derivative)
        return 1;
    return x;
}

double hypertan(double x, int derivative)
{
    if (derivative)
    {
        double r = tanh(x);
        return 1 - (r * r);
    }
    return tanh(x);
}

// CREDITS https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus

double rand_gen()
{
    return ((double)rand() + 1) / ((double)RAND_MAX + 1);
}

double rand_normal(double mu, double sigma)
{
    double v1 = rand_gen();
    double v2 = rand_gen();

    double r1 = cos(2 * 3.14 * v2) * sqrt(-2 * log(v1));
    return r1 * sigma + mu;
}

double mean_squared_error(double *y_true, double *y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++)
    {
        double error = y_true[i] - y_pred[i];
        total += error * error;
    }

    return total / size;
}

double mean_absolute_error(double *y_true, double *y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++)
    {
        double error = y_true[i] - y_pred[i];
        total += fabs(error);
    }
    return total / size;
}