#ifndef __OPERATION_HPP__
#define __OPERATION_HPP__

#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_set>

#include "SmartPointer.hpp"

using namespace std;

template<typename T>
class Operation
{
  public:
    T value = 0;
    T gradient = 0;
    T acc = 0;
    vector<SmartPointer<Operation<double>>> inputs;

    Operation()
      : inputs(0)
    {
    }

    Operation(T value)
      : value(value)
      , inputs(0)
    {
    }

    Operation(SmartPointer<Operation<T>> x)
      : inputs(1)
    {
        inputs[0] = x;
    }

    Operation(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : inputs(2)
    {
        inputs[0] = x;
        inputs[1] = y;
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

    // virtual void clear()
    // {
    //     for (auto& o : this->inputs) {
    //         delete o;
    //     }
    // }

    // Debug
    virtual void print(int depth = 0)
    {
        // Print indentation for better visualization
        for (int i = 0; i < depth; i++) {
            cout << " ";
        }

        // Print node information
        cout << typeid(*this).name() << " value: " << this->value << ", Gradient: " << this->gradient << endl;
        // Recursively print children
        for (auto& o : inputs) {
            o->print(depth + 1);
        }
    }


    virtual ~Operation() {}
};

#endif // __OPERATION_HPP__