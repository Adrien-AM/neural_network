#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <vector>

#define MIN_SOFTMAX_OUTPUT 0.1

void
print_vector(std::vector<double> vec);

void
print_softmax_output(std::vector<double> vec);

#endif // __UTILS_HPP__
