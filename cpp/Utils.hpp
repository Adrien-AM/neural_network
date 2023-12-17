#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <math.h>

#include "Tensor.hpp"

#define MIN_SOFTMAX_OUTPUT 0.1

using namespace std;

void
print_vector(vector<size_t> vec);

void
print_softmax_output(Tensor<double> vec);

Tensor<double>
pad_2d(const Tensor<double>& x, size_t size, double value = 0);

Tensor<double>
convolution_2d(const Tensor<double>& x, const Tensor<double>& k, const double& bias, size_t stride);

#endif // __UTILS_HPP__
