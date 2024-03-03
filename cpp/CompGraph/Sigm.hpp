#ifndef __SIGM_HPP__
#define __SIGM_HPP__

#include "Operation.hpp"

template<typename T>
class Sigm : public Operation<T>
{
  public:
    Sigm(SmartPointer<Operation<T>> x)
      : Operation<T>(x)
    {
    }

    void forward() { this->value = 1 / (1 + exp(-this->inputs[0]->value)); }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->value * (1 - this->value);
    }

    Sigm<T>* copy()
    {
        Sigm<T>* n = new Sigm<T>(this->inputs[0]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __SIGM_HPP__