#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <iostream>
#include <random>
#include <vector>

#include "Activation.hpp"
#include "Layer.hpp"

class Dense : public Layer
{
    std::vector<double> values;
    std::vector<double> delta_errors;
    std::vector<double> biases;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> updates;

    const Activation& activation;

  public:
    Dense(unsigned int layer_size, const Activation& act, bool use_bias = true);
    void init(unsigned int);
    void forward(const std::vector<double>&);
    void backprop(Layer* l, double, double);
    void summarize() const;
    
    unsigned int size() const;
    void reset_values();
    void reset_errors();

    void print_layer() const;

    ~Dense();
};

#endif // __DENSE_HPP__