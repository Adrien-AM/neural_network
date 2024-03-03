#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include "Tensor.hpp"
#include "CompGraph/CompGraph.hpp"

using namespace std;

class Loss
{
  protected:

  public:
    Tensor<double> result;
    Loss() : result() {}
    virtual double evaluate(const Tensor<double> &real, const Tensor<double> &predicted) = 0;
    virtual void backward();
    virtual ~Loss();
};

class MeanAbsoluteError : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
};

class MeanSquaredError : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
};

class CategoricalCrossEntropy : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
};

#endif // __LOSS_HPP__