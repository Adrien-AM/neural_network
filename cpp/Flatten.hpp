#ifndef __FLATTEN_HPP__
#define __FLATTEN_HPP__

#include "Layer.hpp"
#include "Tensor.hpp"
#include <vector>

using namespace std;

class Flatten : public Layer
{
    private:
      vector<size_t> input_shape;
      size_t output_size;

    public:
      void init(vector<size_t>);
      Tensor<double> forward(const Tensor<double>&) const;
      vector<size_t> output_shape() const;
      void summarize() const;
      Flatten* clone() const;
      // debug
      void print_layer() const;
};

#endif