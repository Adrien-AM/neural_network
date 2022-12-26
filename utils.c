#include "utils.h"

void
print_vector(double* v, size_t size)
{
    printf("[ ");
    for (size_t i = 0; i < size - 1; i++) {
        printf("%f ", v[i]);
    }
    printf("%f ]", v[size - 1]);
}

double
relu(double x, int derivative)
{
    if (derivative)
        return x >= 0 ? 1 : 0;
    return x >= 0 ? x : 0;
}

double
sigmoid(double x, int derivative)
{
    if (derivative) {
        double out = sigmoid(x, 0);
        return out * (1 - out);
    }

    return 1 / (1 + exp(-x));
}

double
linear(double x, int derivative)
{
    if (derivative)
        return 1;
    return x;
}

double
hypertan(double x, int derivative)
{
    if (derivative) {
        double r = tanh(x);
        return 1 - (r * r);
    }
    return tanh(x);
}

double*
softmax(double* x, int size)
{
    // allocate memory for the output vector
    double* y = (double*)malloc(sizeof(double) * size);
    (void)x;
    // TODO

    return y;
}

// CREDITS
// https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus

double
rand_gen()
{
    return ((double)rand() + 1) / ((double)RAND_MAX + 1);
}

double
rand_normal(double mu, double sigma)
{
    double v1 = rand_gen();
    double v2 = rand_gen();

    double r1 = cos(2 * 3.14 * v2) * sqrt(-2 * log(v1));
    return r1 * sigma + mu;
}

// Mean Squared Error evaluation
double
mse_f(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += error * error;
    }

    return total / size;
}

// Mean Squared Error derivative
double*
mse_d(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;

    double* result = malloc(sizeof(double) * size);
    for (size_t i = 0; i < size; i++) {
        result[i] = 2 * (y_pred[i] - y_true[i]);
    }

    return result;
}

// Mean Absolute Error evaluation
double
mae_f(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += fabs(error);
    }
    return total / size;
}

// Mean Absolute Error derivative
double*
mae_d(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;

    double* result = malloc(sizeof(double) * size);

    for (size_t i = 0; i < size; i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

// Categorical Cross Entropy evaluation
double
cce_f(double* y_true, double* y_pred, size_t size)
{
    double loss = 0;
    for (size_t i = 0; i < size; i++) {
        loss -= (y_true[i] * log(y_pred[i])) + ((1 - y_true[i]) * log(1 - y_pred[i]));
    }

    return loss / size;
}

// Categorical Cross Entropy derivative
double *
cce_d(double* y_true, double* y_pred, size_t size)
{
    double* result = malloc(sizeof(double) * size);

    for (size_t i = 0; i < size; i++) {
        // Hope none of the denominators is 0
        result[i] = (-y_true[i] / y_pred[i]) + (1 - y_true[i]) / (1 - y_pred[i]);
    }

    return result;
}

const struct loss mean_squared_error = { &mse_f, &mse_d };
const struct loss mean_absolute_error = { &mae_f, &mae_d };
const struct loss categorical_cross_entropy = { &cce_f, &cce_d };