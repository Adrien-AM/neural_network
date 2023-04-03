#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include "Tensor.hpp"

using namespace std;

class Loss
{
  public:
    double (*evaluate)(Tensor<double> real, Tensor<double> predicted);
    Tensor<double> (*derivate)(Tensor<double> real, Tensor<double> predicted);

    Loss(double (*evaluate)(Tensor<double>, Tensor<double>),
         Tensor<double> (*derivate)(Tensor<double>, Tensor<double>))
      : evaluate(evaluate)
      , derivate(derivate){};
};

Loss
MeanAbsoluteError();

Loss
MeanSquaredError();

Loss
CategoricalCrossEntropy();

#endif // __LOSS_HPP__