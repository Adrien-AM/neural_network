#ifndef __EXP_HPP__
#define __EXP_HPP__

#include "Operation.hpp"
#include <math.h>

template <typename T>
class Exp : public Operation<double>
{
private:

public:
  Exp(SmartPointer<Operation<T>> x)
      : Operation<double>(x)
  {
  }

  void forward()
  {
    this->value = exp(this->inputs[0]->value);
  }

  void backward()
  {
    this->inputs[0]->gradient += this->gradient * this->value;
  }
};

#endif // __EXP_HPP__