#ifndef __RESHAPE_HPP__
#define __RESHAPE_HPP__

#include "Layer.hpp"
#include "Tensor.hpp"
#include <vector>

using namespace std;

class Reshape : public Layer
{
  private:
    vector<size_t> input_shape;
    vector<size_t> output_shape;

  public:
    Reshape(vector<size_t> new_shape) : output_shape(new_shape) {}
    void init(vector<size_t>);
    void forward(const Tensor<double>&);
    void backprop(Layer*, double, double);
    void summarize() const;
    // getters and setters
    size_t size() const;
    void reset_values();
    void reset_errors();
    // debug
    void print_layer() const;
};

#endif // __RESHAPE_HPP__