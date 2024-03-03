#ifndef __ADD_HPP__
#define __ADD_HPP__

#include "Operation.hpp"

template<typename T>
class Add : public Operation<T>
{
  public:
    Add(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = this->inputs[0]->value + this->inputs[1]->value; }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient;
        this->inputs[1]->gradient += this->gradient;
    }

    Add<T>* copy()
    {
        Add<T>* n = new Add<T>(this->inputs[0], this->inputs[1]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __ADD_HPP__