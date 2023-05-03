#ifndef __SUB_HPP__
#define __SUB_HPP__

#include "Operation.hpp"

template<typename T>
class Sub : public Operation<T>
{
  public:
    Sub(Operation<T>* x, Operation<T>* y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = this->inputs[0]->value - this->inputs[1]->value; }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient;
        this->inputs[1]->gradient += -this->gradient;
    }
};

#endif // __SUB_HPP__