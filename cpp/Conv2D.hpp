#ifndef __CONV2D_HPP__
#define __CONV2D_HPP__

#include "Layer.hpp"
#include <random>
#include <vector>

using namespace std;

class Conv2D : public Layer
{
  private:
    unsigned int filters_size;
    unsigned int kernel_size;
    unsigned int input_width;
    unsigned int depth;
    unsigned int padding;
    const Activation& activation;
    vector<vector<double>> weights;
    vector<vector<double>> updates;
    vector<double> biases;
    vector<double> values;
    vector<double> padded_input;

  public:
    Conv2D(unsigned int filters,
           unsigned int kernel_size,
           unsigned int padding,
           const Activation& act,
           bool use_bias = true);
    void init(unsigned int);
    void forward(const vector<double>&);
    void backprop(Layer*, double, double);

    void summarize() const;
    unsigned int size() const;
    void reset_values();
    void reset_errors();
    void print_layer() const;

    ~Conv2D();
};

#endif