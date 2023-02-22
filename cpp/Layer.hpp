#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include <iostream>
#include <vector>
#include "Activation.hpp"
#include "Utils.hpp"

class Layer
{
  protected:
    const Activation& activation;

  public:
    std::vector<double> values;
    std::vector<double> actv_values;
    std::vector<double> errors;
    std::vector<double> delta_errors;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(const Activation& act, unsigned int layer_size, bool use_bias);

    // abstraction
    virtual void forward(std::vector<double>) = 0;
    virtual void backprop(Layer *, double) = 0;
    virtual void init(unsigned int) = 0;
    virtual void summarize() = 0;

    // getters and setters
    unsigned int size();

    // debug
    void print_layer() const;

    virtual ~Layer();
};

#endif // __LAYER_HPP__