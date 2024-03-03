#ifndef __MAX_HPP__
#define __MAX_HPP__

#include "Operation.hpp"

template<typename T>
class Max : public Operation<T>
{
  public:
    Max(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = max(this->inputs[0]->value, this->inputs[1]->value); }

    void backward()
    {
        this->inputs[0]->gradient += this->inputs[0]->value == this->value ? this->gradient : 0;
        this->inputs[1]->gradient += this->inputs[1]->value == this->value ? this->gradient : 0;
    }

    Max<T>* copy()
    {
        Max<T>* n = new Max<T>(this->inputs[0], this->inputs[1]);
        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __MAX_HPP__