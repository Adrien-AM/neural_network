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
    /**
     * Initialize the layer. Called when the full neural network architecture is known.
     * 
     * @param input_shape : vector<size_t>
    */
    virtual void init(vector<size_t>) = 0;

    /**
     * Forward pass through the layer. Result is computed and stored in `output_values`.
     * 
     * @param input : input data (most of the time, output of previous layer). input should be the same shape as specified in init().
    */
    virtual void forward(const Tensor<double>&) = 0;

    /**
     * Backward pass through the layer. Downstream gradients are written on previous layer's `errors`.
     * 
     * @param input_layer
     * @param learning_rate : hyperparameter
     * @param momentum : hyperparameter
    */
    virtual void backprop(Layer*, double, double) = 0;

    /**
     * Displays useful informations on stdout.
    */
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