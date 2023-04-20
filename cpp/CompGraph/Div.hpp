#ifndef __DIV_HPP__
#define __DIV_HPP__

#include "Operation.hpp"

template<typename T>
class Div : public Operation<T>
{
  public:
    Div(Operation<T>* x, Operation<T>* y)
      : Operation<T>(x, y)
    {
    }

    void forward() { this->value = this->inputs[0]->value / this->inputs[1]->value; }

    void backward()
    {
        this->inputs[0]->gradient += this->gradient * (1 / this->inputs[1]->value);
        this->inputs[1]->gradient += this->gradient * -this->inputs[0]->value /
                                    (this->inputs[1]->value * this->inputs[1]->value);
    }
};

#endif // __DIV_HPP__