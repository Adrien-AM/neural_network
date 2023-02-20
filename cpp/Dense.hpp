#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <vector>

#include "Layer.hpp"
#include "Neuron.hpp"

class Dense : public Layer
{
  private:
    double (*const activation)(double, int);
    std::vector<Neuron *> neurons;

  public:
    Dense(double (*const activation)(double, int), std::vector<Neuron *> neurons)
      : activation(activation), neurons(neurons) {
    };
    Dense(const Dense& d) : activation(d.activation), neurons(d.neurons) {};
    void forward();
    void backprop();
    ~Dense();
};

#endif // __DENSE_HPP__