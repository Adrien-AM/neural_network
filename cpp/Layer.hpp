#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "Activation.hpp"
#include "Utils.hpp"
#include <iostream>


using namespace std;

class Layer
{
  public:
    Tensor<double> output_values;
    Tensor<double> errors;

    // abstraction
    virtual void init(vector<size_t>) = 0;
    virtual void forward(const Tensor<double>&) = 0;
    virtual void backprop(Layer*, double, double) = 0;
    virtual void summarize() const = 0;
    // getters and setters
    virtual size_t size() const = 0;
    virtual void reset_values() = 0;
    virtual void reset_errors() = 0;
    // debug
    virtual void print_layer() const = 0;

    virtual ~Layer() = 0;
};

#endif // __LAYER_HPP__