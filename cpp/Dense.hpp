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
  private:
    size_t size;
    vector<size_t> shape;

  public:
    
    Dense(size_t layer_size, bool use_bias = true);
    void init(vector<size_t>);
    Tensor<double> forward(const Tensor<double>&) const;
    vector<size_t> output_shape() const;
    void summarize() const;
  
    void print_layer() const;
    Dense* clone() const;

    ~Dense();
};

#endif // __DENSE_HPP__