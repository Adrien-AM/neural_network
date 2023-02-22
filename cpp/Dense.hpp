#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <vector>
#include <iostream>
#include <random>

#include "Layer.hpp"
#include "Activation.hpp"


class Dense : public Layer
{
  public:
    Dense(const Activation& act, unsigned int layer_size, bool use_bias = true);
    void forward(std::vector<double>);
    void backprop(Layer *l, double);
    void init(unsigned int);
    void summarize();
};

#endif // __DENSE_HPP__