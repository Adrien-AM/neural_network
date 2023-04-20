#ifndef __ABS_HPP__
#define __ABS_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Abs : public Operation<T>
{
  private:

  public:
    Abs(Operation<T>* x)
      : Operation<T>(x)
    {
    }

    void forward()
    {
        this->value = abs(this->inputs[0]->value);
    }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * (this->value >= 0);
    }
};

#endif // __ABS_HPP__