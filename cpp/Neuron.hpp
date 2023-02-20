#ifndef __NEURON_HPP__
#define __NEURON_HPP__

class Neuron
{
    private:
      double value;
      double actv_value;
      double error;
      double delta_error;

      

    public:
      Neuron(const Neuron& n) : value(n.value), actv_value(n.actv_value), error(n.error), delta_error(n.delta_error){};
      virtual void feed() = 0;
};

#endif // __NEURON_HPP__