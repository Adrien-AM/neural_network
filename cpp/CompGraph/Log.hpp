#ifndef __LOG_HPP__
#define __LOG_HPP__

#include "Operation.hpp"
#include <math.h>

template<typename T>
class Log : public Operation<T>
{
  private:
    T x;

  public:
    Log(SmartPointer<Operation<T>> x)
      : Operation<T>(x)
    {
    }

    void forward()
    {
        x = this->inputs[0]->value;
        this->value = log(x);
    }

    void backward() { this->inputs[0]->gradient += this->gradient * 1 / x; }

    Log<T>* copy()
    {
        Log<T>* n = new Log<T>(this->inputs[0]);
        n->value = this->value;
        n->gradient = this->gradient;
        n->x = this->x;
        return n;
    }
};

#endif // __LOG_HPP__