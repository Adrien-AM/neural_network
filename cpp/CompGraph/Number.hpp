#ifndef __NUMBER_HPP__
#define __NUMBER_HPP__

#include "Operation.hpp"

template<typename T>
class Number : public Operation<T>
{
  public:
    Number(T value)
      : Operation<T>(value)
    {
    }

    Number()
      : Operation<T>(0.0)
    {
    }

    void forward() {}
    void backward() {}
};

#endif // __NUMBER_HPP__
