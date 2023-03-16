#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <vector>
#include <iostream>
#include <random>

#include <omp.h>

#include "Layer.hpp"
#include "Activation.hpp"


class Dense : public Layer
{
  public:
    Dense(unsigned int layer_size, const Activation& act, bool use_bias = true);
    void forward(const std::vector<double>&);
    void backprop(Layer *l, double, double);
    void init(unsigned int);
    void summarize();
};

#endif // __DENSE_HPP__