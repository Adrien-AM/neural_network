#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Prints a vector : [ a, b, c ] (no newline)
void
print_vector(double* v, size_t size);

void
print_softmax(double* result, size_t nb_classes, double until);

// Activation functions : takes neuron output z=wx+b, returns final neuron output
// if second parameter is set to true, it will return the function's derivative
double
relu(double, int);
double
sigmoid(double, int);
double
linear(double, int);
double
hypertan(double, int);

// Random number from normal function with parameters mu and sigma
double
rand_normal(double, double);

/*
 * @var evaluate : (y_true, y_pred, size) : takes the real output and prediction, returns a single
 * value of loss
 * @var derivative : (y_true, y_pred, size) : takes the real output and prediction, returns a vector
 * of loss derivated with respect to the output. don't forget to free !
 */
struct loss
{
    double (*evaluate)(double*, double*, size_t);
    double* (*derivative)(double*, double*, size_t);
};

// Loss functions
extern const struct loss mean_squared_error;
extern const struct loss mean_absolute_error;
extern const struct loss cross_entropy;

#endif