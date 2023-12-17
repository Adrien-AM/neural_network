#include <iostream>
#include <fstream>

#include "Loss.hpp"
#include "Utils.hpp"

// AUTODIFF
void
Loss::backward()
{
    CompGraph<double> graph = this->result.get_graph();
    graph.backward();
}

Loss::~Loss() {}

double
MeanAbsoluteError::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    result = (y_true - y_pred).abs().sum() / (double)y_true.size();
    return result;
}

double
MeanSquaredError::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    result = (y_true - y_pred).pow(2.0).sum() / (double)y_true.size();
    return result;
}

double
CategoricalCrossEntropy::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    result = -(y_true * y_pred.log()).sum();
    return (double)result;
}
