#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <vector>
#include <math.h>

#define MIN_SOFTMAX_OUTPUT 0.1

void
print_vector(std::vector<double> vec);

void
print_softmax_output(std::vector<double> vec);

std::vector<double> convolution_product(const std::vector<double>&, const std::vector<double>&, unsigned int, unsigned int);

#endif // __UTILS_HPP__

