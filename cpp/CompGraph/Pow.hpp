#ifndef __POW_HPP__
#define __POW_HPP__

#include "Operation.hpp"
#include <math.h>

template <typename T>
class Pow : public Operation<double>
{
private:
  T x;
  T p;

public:
  Pow(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<double>(x, y)
  {
  }

  void forward()
  {
    x = this->inputs[0]->value;
    p = this->inputs[1]->value;
    this->value = pow(x, p);
  }

  void backward()
  {
    this->inputs[0]->gradient += this->gradient * (p * pow(x, p - 1));
    this->inputs[1]->gradient += this->value * log(x);
  }
};

#endif // __POW_HPP__