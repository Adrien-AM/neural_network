#include "Activation.hpp"
#include "Utils.hpp"

// -- Linear --

Tensor<double>
Linear::compute(const Tensor<double>& x) const
{
    return x;
}

// -- ReLU --

Tensor<double>
ReLU::compute(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.max(0);
    return result;
}

// -- Sigmoid --

Tensor<double>
Sigmoid::compute(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.sigm();
    return result;
}

// -- Softmax --

Tensor<double>
Softmax::compute(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.exp();
    result /= result.sum();
    return result;
}