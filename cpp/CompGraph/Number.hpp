#ifndef __NUMBER_HPP__
#define __NUMBER_HPP__

#include "Operation.hpp"

template<typename T>
class Number : public Operation<T>
{
  public:
    size_t nb = 0;
    Number(T value)
      : Operation<T>(value)
    {
    }

    Number(SmartPointer<Operation<T>> input)
      : Operation<T>(input)
    {
        this->value = input->value;
    }

    Number()
      : Operation<T>(0.0)
    {
    }

    void forward() {}
    void backward()
    {
        this->nb++;
        // if (nb>1) printf("Multiple backward :(\n");
        for (auto& i : this->inputs) {
            i->gradient += this->gradient;
        }
    }

    Number<T>* copy()
    {
        Number<T>* n = new Number<T>();
        if (!this->inputs.empty()) {
            n->inputs = vector<SmartPointer<Operation<T>>>(1);
            n->inputs[0] = this->inputs[0];
        }

        n->value = this->value;
        n->gradient = this->gradient;
        return n;
    }
};

#endif // __NUMBER_HPP__
