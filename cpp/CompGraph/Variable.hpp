#ifndef __VARIABLE_HPP__
#define __VARIABLE_HPP__

#include "Operation.hpp"

template<typename T>
class Variable : public Operation<T>
{
  public:
    Variable(T value)
      : Operation<T>(value)
    {
    }
    
    void forward() {}
    void backward() {}
};

#endif // __VARIABLE_HPP__
