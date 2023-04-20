#ifndef __OPERATION_HPP__
#define __OPERATION_HPP__

#include <vector>

using namespace std;

template<typename T>
class Operation
{
  public:
    vector<Operation*> inputs;
    T value;
    T gradient;

    Operation()
      : inputs(0)
      , value(0)
      , gradient(0)
    {
    }

    Operation(T value)
      : inputs(0)
      , value(value)
      , gradient(0)
    {
    }

    Operation(Operation* x)
      : value(0)
      , gradient(0)
    {
        add_input(x);
    }

    Operation(Operation* x, Operation* y)
      : value(0)
      , gradient(0)
    {
        add_input(x);
        add_input(y);
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

    void add_input(Operation* o) { inputs.push_back(o); }

    virtual ~Operation() {}
};

#endif // __OPERATION_HPP__