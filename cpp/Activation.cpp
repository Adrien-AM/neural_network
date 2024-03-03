#include "Activation.hpp"
#include "Utils.hpp"

void
Activation::init(vector<size_t> shape)
{
    this->shape = shape;
}

vector<size_t>
Activation::output_shape() const
{
    return this->shape;
}

Activation::~Activation() {}

// -- Linear --

Tensor<double>
Linear::forward(const Tensor<double>& x) const
{
    return x;
}

Linear*
Linear::clone() const
{
    Linear* cloned = new Linear();
    cloned->init(this->shape);
    return cloned;
}

// -- ReLU --

Tensor<double>
ReLU::forward(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.max(0);
    return result;
}

ReLU*
ReLU::clone() const
{
    ReLU* cloned = new ReLU();
    cloned->init(this->shape);
    return cloned;
}

// -- Sigmoid --

Tensor<double>
Sigmoid::forward(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.sigm();
    return result;
}

Sigmoid*
Sigmoid::clone() const
{
    Sigmoid* cloned = new Sigmoid();
    cloned->init(this->shape);
    return cloned;
}

// -- Softmax --

Tensor<double>
Softmax::forward(const Tensor<double>& x) const
{
    Tensor<double> result(x.shape());
    result = x.exp();
    result /= result.sum();
    return result;
}

Softmax*
Softmax::clone() const
{
    Softmax* cloned = new Softmax();
    cloned->init(this->shape);
    return cloned;
}