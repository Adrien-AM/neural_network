#ifndef __ABS_HPP__
#define __ABS_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Abs : public Operation<double>
{
  private:
    T sign;

  public:
    Abs(SmartPointer<Operation<T>> x)
      : Operation<double>(x)
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
};

#endif // __ABS_HPP__