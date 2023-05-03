#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include "Layer.hpp"


using namespace std;

class Input : public Layer
{
  public:
    Input(Tensor<double>);
    void init(vector<size_t>);
    void forward(const Tensor<double>&);
    Tensor<double> backprop(Layer*);
    void summarize() const;
    size_t size() const;
    void reset_values();
    void reset_errors();
    void print_layer() const;
};

#endif // __INPUT_HPP__