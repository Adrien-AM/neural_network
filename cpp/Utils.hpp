#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <math.h>

#include "Tensor.hpp"

#define MIN_SOFTMAX_OUTPUT 0.1

using namespace std;

void
print_vector(Tensor<double> vec);

void
print_softmax_output(Tensor<double> vec);

Tensor<double>
add_padding_2d(const Tensor<double>& image, size_t pad_size);

Tensor<double>
convolution_2d(const Tensor<double>& input, const Tensor<double>& kernel, size_t stride);

Tensor<double>
convolution_product(const Tensor<double>& input, const Tensor<double>& filter, size_t stride);

#endif // __UTILS_HPP__
