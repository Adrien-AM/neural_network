#ifndef __CONV2D_HPP__
#define __CONV2D_HPP__

#include "Layer.hpp"
#include <random>
#include <vector>

class Conv2D : public Layer
{
  private:
    unsigned int filters_size;
    unsigned int kernel_size;
    unsigned int input_width;
    unsigned int padding;
    const Activation& activation;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> updates;
    std::vector<double> biases;
    std::vector<double> values;
    std::vector<double> padded_input;

  public:
    Conv2D(unsigned int filters,
           unsigned int kernel_size,
           unsigned int input_width,
           unsigned int padding,
           const Activation& act,
           bool use_bias = true);
    void init(unsigned int);
    void forward(const std::vector<double>&);
    void backprop(Layer*, double, double);

    void summarize() const;
    unsigned int size() const;
    void reset_values();
    void reset_errors();
    void print_layer() const;

    ~Conv2D();
};

#endif