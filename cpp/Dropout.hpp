#ifndef __DROPOUT_HPP__
#define __DROPOUT_HPP__

#include <random>
#include "Layer.hpp"

using namespace std;

class Dropout : public Layer
{
  private:
    double rate;
    vector<bool> actives;

  public:
    Dropout(double rate);

    void init(vector<size_t>);

    void forward(const Tensor<double>&);
    Tensor<double> backprop(Layer*);
    void summarize() const;

    size_t size() const;
    void reset_values();
    void reset_errors();

    // debug
    void print_layer() const;

    ~Dropout();
};

#endif // __DROPOUT_HPP__