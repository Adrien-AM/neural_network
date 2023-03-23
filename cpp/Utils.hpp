#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <math.h>
#include <vector>

#define MIN_SOFTMAX_OUTPUT 0.1

using namespace std;

void
print_vector(vector<double> vec);

void
print_softmax_output(vector<double> vec);

vector<double>
add_padding(const vector<double>& image, unsigned int width, unsigned int pad_size);

vector<double>
convolution_product(const vector<double>& input,
                    const vector<double>& filter,
                    unsigned int width,
                    unsigned int stride);

#endif // __UTILS_HPP__
