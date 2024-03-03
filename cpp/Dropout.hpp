#ifndef __DROPOUT_HPP__
#define __DROPOUT_HPP__

#include <random>
#include "Layer.hpp"

using namespace std;

class Dropout : public Layer
{
  private:
    double rate;
    vector<size_t> shape;

  public:
    Dropout(double rate);

    void init(vector<size_t>);

    Tensor<double> forward(const Tensor<double>&) const;
    vector<size_t> output_shape() const;
    void summarize() const;

    // debug
    void print_layer() const;

    ~Dropout();
};

#endif // __DROPOUT_HPP__