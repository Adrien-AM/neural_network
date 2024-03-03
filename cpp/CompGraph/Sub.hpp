#ifndef __SUB_HPP__
#define __SUB_HPP__

#include "Operation.hpp"

template<typename T>
class Sub : public Operation<T>
{
  public:
    Sub(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = this->inputs[0]->value - this->inputs[1]->value; }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient;
        this->inputs[1]->gradient -= this->gradient;
    }

    Sub<T>* copy()
    {
        Sub<T>* n = new Sub<T>(this->inputs[0], this->inputs[1]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __SUB_HPP__