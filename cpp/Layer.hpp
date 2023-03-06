#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "Activation.hpp"
#include "Utils.hpp"
#include <iostream>
#include <vector>

class Layer
{
  protected:

  public:
    std::vector<double> output_values;
    std::vector<double> errors;

    // abstraction
    virtual void init(unsigned int) = 0;
    virtual void forward(const std::vector<double>&) = 0;
    virtual void backprop(Layer*, double, double) = 0;
    virtual void summarize() const = 0;
    // getters and setters
    virtual unsigned int size() const = 0;
    // debug
    virtual void print_layer() const = 0;

    virtual ~Layer() = 0;
};

#endif // __LAYER_HPP__