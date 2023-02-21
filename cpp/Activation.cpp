#include <cmath>
#include "Activation.hpp"

// -- Linear --

double
Linear::compute(double x) const
{
    return x;
}

double
Linear::derivative(double x) const
{
    return 1;
}

// -- ReLU --

double
ReLU::compute(double x) const
{
    return x > 0 ? x : 0;
}

double
ReLU::derivative(double x) const
{
    return x > 0 ? 1 : 0;
}

// -- Sigmoid --

double
Sigmoid::compute(double x) const
{
    return 1/(1 + exp(-x));
}

double
Sigmoid::derivative(double x) const
{
    double sigma = this->compute(x);
    return sigma * (1 - sigma);
}

// -- Softmax --
// TODO