#ifndef __CONV2D_HPP__
#define __CONV2D_HPP__

#include "Layer.hpp"
#include <random>


using namespace std;

class Conv2D : public Layer
{
  private:
    size_t filters_size;
    size_t kernel_size;
    size_t channels;
    size_t padding;
    Tensor<double> padded_input;

  public:
    Conv2D(size_t filters,
           size_t kernel_size,
           size_t padding,
           bool use_bias = true);
    void init(vector<size_t>);
    Tensor<double> forward(const Tensor<double>&);

    void summarize() const;
    size_t size() const;
    void print_layer() const;
    Conv2D* clone() const;

    ~Conv2D();
};

#endif