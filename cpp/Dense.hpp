#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <iostream>
#include <random>
#include <vector>

#include "Activation.hpp"
#include "Layer.hpp"

using namespace std;

class Dense : public Layer
{
    vector<double> values;
    vector<double> delta_errors;
    vector<double> biases;
    vector<vector<double>> weights;
    vector<vector<double>> updates;

    const Activation& activation;

  public:
    Dense(unsigned int layer_size, const Activation& act, bool use_bias = true);
    void init(unsigned int);
    void forward(const vector<double>&);
    void backprop(Layer* l, double, double);
    void summarize() const;
    
    unsigned int size() const;
    void reset_values();
    void reset_errors();

    void print_layer() const;

    ~Dense();
};

#endif // __DENSE_HPP__