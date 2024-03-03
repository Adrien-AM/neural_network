#ifndef __ABS_HPP__
#define __ABS_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Abs : public Operation<T>
{
  private:
    T sign;

  public:
    Abs(SmartPointer<Operation<T>> x)
      : Operation<T>(x)
    {
    }

    void forward()
    {
        sign = this->inputs[0]->value >= 0 ? 1 : -1;
        this->value = abs(this->inputs[0]->value);
    }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->sign;
    }

    Abs<T>* copy()
    {
        Abs<T>* n = new Abs<T>(this->inputs[0]);
        n->value = this->value;
        n->gradient = this->gradient;
        n->sign = sign;
        return n;
    }
};

#endif // __ABS_HPP__