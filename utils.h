#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Prints a vector : [ a, b, c ] (no newline)
void
print_vector(double* v, size_t size);

// Activation functions : takes neuron output z=wx+b, returns final neuron output
// if second parameter is set to true, it will return the function's derivative
double relu(double, int);
double sigmoid(double, int);
double linear(double, int);
double hypertan(double, int);

double*
softmax(double*, int);

// Random number from normal function with parameters mu and sigma
double rand_normal(double, double);

// Loss functions
double mean_squared_error(double *y_true, double *y_pred, size_t size);
double mean_absolute_error(double *y_true, double *y_pred, size_t size);
double
categorical_cross_entropy(double* y_true, double* y_pred, size_t size);

#endif