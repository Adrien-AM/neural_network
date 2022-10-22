#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Activation functions : takes neuron output z=wx+b, returns final neuron output
// if second parameter is set to true, it will return the function's derivative
float relu(float, int);
float sigmoid(float, int);
float linear(float, int);

// Random number from normal function with parameters mu and sigma
float rand_normal(float, float);

// Loss functions
float mean_squared_error(float *y_true, float *y_pred, size_t size);

// Data functions
void generate_data_inputs(size_t data_size, size_t input_size,
                          float **inputs, int range_start, int range_end);
void generate_data_outputs(size_t data_size, size_t output_size, float ** inputs,
                           float **outputs, float (*func)(float *));
float **read_csv(char *filename, size_t *nb_lines, size_t *nb_columns, char **columns);

#endif