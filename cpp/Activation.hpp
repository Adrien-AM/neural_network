#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <cmath>
#include <iostream>

#include "Layer.hpp"
#include "Tensor.hpp"

using namespace std;

class Activation : public Layer
{
  protected:
    vector<size_t> shape;

  public:
    void init(vector<size_t>);
    vector<size_t> output_shape() const;
    ~Activation();
};

class Linear : public Activation
{
  public:
    Tensor<double> forward(const Tensor<double>&) const;
    Linear* clone() const;
};

class ReLU : public Activation
{
  public:
    Tensor<double> forward(const Tensor<double>&) const;
    ReLU* clone() const;
};

class Sigmoid : public Activation
{
  public:
    Tensor<double> forward(const Tensor<double>&) const;
    Sigmoid* clone() const;
};

class Softmax : public Activation
{
  public:
    Tensor<double> forward(const Tensor<double>&) const;
    Softmax* clone() const;
};

#endif // __ACTIVATION_H__