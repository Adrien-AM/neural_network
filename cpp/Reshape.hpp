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
    vector<size_t> shape;

  public:
    Reshape(vector<size_t> new_shape) : output_shape(new_shape) {}
    void init(vector<size_t>);
    Tensor<double> forward(const Tensor<double>&) const;
    vector<size_t> output_shape() const;
    void summarize() const;
    // getters and setters
    Reshape* clone() const;
    // debug
    void print_layer() const;
};

#endif // __RESHAPE_HPP__