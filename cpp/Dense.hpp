#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <iostream>
#include <random>


#include "Activation.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"

using namespace std;


/**
 * Careful : Dense should only be used if input is 1D ! (for now)
*/
class Dense : public Layer
{
    Tensor<double> values;
    Tensor<double> delta_errors;
    Tensor<double> biases;

    const Activation& activation;

  public:
    Dense(size_t layer_size, const Activation& act, bool use_bias = true);
    void init(vector<size_t>);
    void forward(const Tensor<double>&);
    Tensor<double> backprop(Layer*);
    void summarize() const;
    
    size_t size() const;
    void reset_values();
    void reset_errors();

    void print_layer() const;

    ~Dense();
};

#endif // __DENSE_HPP__