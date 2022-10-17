#ifndef __UTILS_H__
#define __UTILS_H__

// Activation functions : takes neuron output z=wx+b, returns final neuron output
// if second parameter is set to true, it will return the function's derivative
float relu(float, int);
float sigmoid(float, int);
float linear(float, int);

// Random number from normal function with parameters mu and sigma
float rand_normal(float, float);

#endif