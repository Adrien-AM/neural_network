#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include "Layer.hpp"
#include <vector>

using namespace std;

class Input : public Layer
{
  public:
    Input(vector<double>);
    void init(unsigned int);
    void forward(const vector<double>&);
    void backprop(Layer*, double, double);
    void summarize() const;
    unsigned int size() const;
    void reset_values();
    void reset_errors();
    void print_layer() const;
};

#endif // __INPUT_HPP__