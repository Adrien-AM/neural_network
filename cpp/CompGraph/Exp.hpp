#ifndef __EXP_HPP__
#define __EXP_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Exp : public Operation<T>
{
  private:
    T x;

  public:
    Exp(Operation<T>* x)
      : Operation<T>(x)
    {
    }

    void forward()
    {
        x = this->inputs[0]->value;
        this->value = exp(x);
    }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->value;
    }
};

#endif // __EXP_HPP__