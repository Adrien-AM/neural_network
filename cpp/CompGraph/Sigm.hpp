#ifndef __SIGM_HPP__
#define __SIGM_HPP__

#include "Operation.hpp"

template<typename T>
class Sigm : public Operation<double>
{
  public:
    Sigm(SmartPointer<Operation<T>> x)
      : Operation<double>(x)
    {
    }

    void forward() { this->value = 1 / (1 + exp(-this->inputs[0]->value)); }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->value * (1 - this->value);
    }
};

#endif // __SIGM_HPP__