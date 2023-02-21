#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <vector>
#include "Layer.hpp"

class Input : public Layer
{
    public:
      Input(std::vector<double>);
      void forward(std::vector<double>);
      void backprop(Layer*, double);
      void init(unsigned int);

};

#endif // __INPUT_HPP__