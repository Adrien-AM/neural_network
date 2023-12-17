#ifndef __OPTIMIZER_HPP__
#define __OPTIMIZER_HPP__

#include "Layer.hpp"

class Optimizer
{
  protected:
    vector<Layer*> layers;
    double alpha;
    double clip;

  public:
    Optimizer(double lr = 1e-3, double clip = 0) : alpha(lr), clip(clip) {}
    virtual void attach_layers(vector<Layer*> l) { layers = l; }
    virtual void update(size_t batch_size) = 0;
    void sample();
};

class SGD : public Optimizer
{
  public:
    SGD(double alpha = 1e-3, double clip = 0)
      : Optimizer(alpha, clip)
    {
    }
    void update(size_t batch_size);
};

class Adam : public Optimizer
{
    private:
      double beta1, beta2;
      vector<Tensor<double>> updates1;
      vector<Tensor<double>> updates2;
      double t;

    public:
      Adam(double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double clip = 0);
      void attach_layers(vector<Layer*> l);
      void update(size_t batch_size);
};

#endif // __OPTIMIZER_HPP__
