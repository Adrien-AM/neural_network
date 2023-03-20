#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <math.h>
#include <vector>

#define MIN_SOFTMAX_OUTPUT 0.1

void
print_vector(std::vector<double> vec);

void
print_softmax_output(std::vector<double> vec);

std::vector<double>
add_padding(std::vector<double>& image, unsigned int width, unsigned int pad_size);

std::vector<double>
convolution_product(const std::vector<double>&,
                    const std::vector<double>&,
                    unsigned int,
                    unsigned int);

#endif // __UTILS_HPP__
