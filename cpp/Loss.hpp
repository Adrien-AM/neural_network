#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include "CompGraph/Add.hpp"
#include "CompGraph/CompGraph.hpp"
#include "CompGraph/Log.hpp"
#include "CompGraph/Mul.hpp"
#include "CompGraph/Pow.hpp"
#include "CompGraph/Sub.hpp"
#include "CompGraph/Variable.hpp"
#include "CompGraph/Div.hpp"
#include "Tensor.hpp"

using namespace std;

class Loss
{
  protected:
    CompGraph<double>* graph;
    Tensor<Variable<double>*> inputs;

    void create_variables(const Tensor<double>& predicted);

  public:
    Loss() : graph(nullptr) {}
    virtual double evaluate(const Tensor<double>& real, const Tensor<double>& predicted) = 0;
    virtual Tensor<double> derivate(const Tensor<double>& real, const Tensor<double>& predicted);
    virtual ~Loss();
};

class MeanAbsoluteError : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
    Tensor<double> derivate(const Tensor<double>& real, const Tensor<double>& predicted);
};

class MeanSquaredError : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
    Tensor<double> derivate(const Tensor<double>& real, const Tensor<double>& predicted);
};

class CategoricalCrossEntropy : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
    Tensor<double> derivate(const Tensor<double>& real, const Tensor<double>& predicted);
};

class SSIM : public Loss
{
  public:
    double evaluate(const Tensor<double>& real, const Tensor<double>& predicted);
};

#endif // __LOSS_HPP__