#ifndef __OPERATION_HPP__
#define __OPERATION_HPP__

#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "SmartPointer.hpp"

using namespace std;

template<typename T>
class Operation
{
  public:
    T value = 0;
    T gradient = 0;
    vector<SmartPointer<Operation<T>>> inputs;

    // Initialize node directly from value
    Operation(T value)
      : value(value)
      , inputs(0)
    {
    }

    // Unary operators
    Operation(SmartPointer<Operation<T>> x)
      : inputs(1)
    {
        inputs[0] = x;
    }


    // Binary operators
    Operation(SmartPointer<Operation<T>> x, SmartPointer<Operation<T>> y)
      : inputs(2)
    {
        inputs[0] = x;
        inputs[1] = y;
    }

    // Create a NON-RECURSIVE copy
    // i.e. a node with same value, gradient, AND connected to the same graph
    virtual Operation<T>* copy() = 0;

    // Node value
    operator T() { return value; }

    // When assigning a value directly, the computation graph is not valid anymore ; detach it.
    void operator=(T v)
    {
        value = v;
        // Detach
        gradient = 0;
        inputs = {};
    }

    // Forward pass of operator should store result in its 'value' attribute.
    virtual void forward() = 0;
    // Backward pass should add gradient to all of its inputs 'gradient' attribute.
    virtual void backward() = 0;

    // Debug pretty print
    virtual void print(int depth = 0)
    {
        // Print indentation for better visualization
        for (int i = 0; i < depth; i++) {
            cout << " ";
        }

        // Print node information
        cout << typeid(*this).name() << " value: " << this->value
             << ", Gradient: " << this->gradient << endl;
        // Recursively print children
        for (auto& o : inputs) {
            o->print(depth + 1);
        }
    }

    virtual ~Operation() {}
};

#endif // __OPERATION_HPP__