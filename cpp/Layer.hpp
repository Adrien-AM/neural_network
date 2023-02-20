#ifndef __LAYER_HPP__
#define __LAYER_HPP__

class Layer {
    public:
      Layer();
      Layer(const Layer& l); // constructor by copy
      virtual void forward() = 0;
      virtual void backprop() = 0;
};

#endif // __LAYER_HPP__