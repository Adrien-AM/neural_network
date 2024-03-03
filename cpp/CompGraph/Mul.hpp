#ifndef __MUL_HPP__
#define __MUL_HPP__

#include "Operation.hpp"

template<typename T>
class Mul : public Operation<T>
{
  public:
    Mul(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = this->inputs[0]->value * this->inputs[1]->value; }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * this->inputs[1]->value;
        this->inputs[1]->gradient += this->gradient * this->inputs[0]->value;
    }

    Mul<T>* copy()
    {
        Mul<T>* n = new Mul<T>(this->inputs[0], this->inputs[1]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __MUL_HPP__