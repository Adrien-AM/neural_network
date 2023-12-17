#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__


#include <cmath>
#include <iostream>

#include "Tensor.hpp"

using namespace std;

class Activation
{
  public:
    virtual Tensor<double> compute(const Tensor<double>&) const = 0;
};

class Linear : public Activation
{
  public:
    Tensor<double> compute(const Tensor<double>&) const;
};

class ReLU : public Activation
{
  public:
    Tensor<double> compute(const Tensor<double>&) const;
};

class Sigmoid : public Activation
{
  public:
    Tensor<double> compute(const Tensor<double>&) const;
};

class Softmax : public Activation
{
  public:
    Tensor<double> compute(const Tensor<double>&) const;
};

#endif // __ACTIVATION_H__