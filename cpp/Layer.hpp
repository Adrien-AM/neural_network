#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "Utils.hpp"
#include <iostream>

using namespace std;

class Layer
{
  public:
    Tensor<double> weights;
    Tensor<double> biases;
    
    /**
     * Initialize the layer. Called when the full neural network architecture is known.
     *
     * @param input_shape : vector<size_t>
     */
    virtual void init(vector<size_t>) = 0;

    /**
     * Forward pass through the layer. Result is computed and stored in `output_values`.
     *
     * @param input : input data (most of the time, output of previous layer). input should be the
     * same shape as specified in init().
     */
    virtual Tensor<double> forward(const Tensor<double>&) const = 0;

    /**
     * Returns shape of the output from the layer.
     */
    virtual vector<size_t> output_shape() const = 0;

    /**
     * Displays useful informations on stdout.
     */
    virtual void summarize() const;

    virtual Layer* clone() const = 0;

    // debug
    virtual void print_layer() const;

    virtual ~Layer() = 0;
};

#endif // __LAYER_HPP__