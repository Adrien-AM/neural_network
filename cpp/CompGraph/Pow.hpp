#ifndef __POW_HPP__
#define __POW_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Pow : public Operation<T>
{
  private:
    T x;
    T p;

  public:
    Pow(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<T>(x, y)
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

    Pow<T>* copy()
    {
        Pow<T>* n = new Pow<T>(this->inputs[0], this->inputs[1]);
        n->value = this->value;
        n->gradient = this->gradient;
        n->x = this->x;
        n->p = this->p;
        return n;
    }
};

#endif // __POW_HPP__