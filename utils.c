#include "utils.h"

void print_vector(double *v, size_t size)
{
    printf("[ ");
    for (size_t i = 0; i < size - 1; i++) {
        printf("%f ", v[i]);
    }
    printf("%f ]", v[size - 1]);
}

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

// Credits : ChatGPT :)
double* softmax(double* x, int size) {
    // allocate memory for the output vector
    double* y = (double*)malloc(sizeof(double) * size);

    // find the maximum value in the input vector
    double max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // compute the numerator and denominator of the softmax function
    double numerator = 0;
    double denominator = 0;
    for (int i = 0; i < size; i++) {
        numerator += exp(x[i] - max);
        denominator += numerator;
    }

    // compute the softmax of the input vector
    for (int i = 0; i < size; i++) {
        y[i] = numerator / denominator;
    }

    return y;
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

double
categorical_cross_entropy(double* y_true, double* y_pred, size_t size)
{
    double loss = 0;
    for (size_t i = 0; i < size; i++) {
        loss -= (y_true[i] * log(y_pred[i])) + ((1 - y_true[i]) * log(1 - y_pred[i]));
    }

    return loss / size;
}