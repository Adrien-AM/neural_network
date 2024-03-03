#ifndef __EXP_HPP__
#define __EXP_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Exp : public Operation<T>
{
  public:
    Exp(SmartPointer<Operation<T>> x)
      : Operation<T>(x)
    {
    }

    void forward() { this->value = exp(this->inputs[0]->value); }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->value;
    }

    Exp<T>* copy()
    {
        Exp<T>* n = new Exp<T>(this->inputs[0]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __EXP_HPP__