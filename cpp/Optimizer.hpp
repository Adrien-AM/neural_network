#ifndef __OPTIMIZER_HPP__
#define __OPTIMIZER_HPP__

#include "Layer.hpp"

class Optimizer
{
  protected:
    vector<Layer*> layers;
    double alpha;

  public:
    Optimizer(double lr = 1e-3) : alpha(lr) {}
    virtual void attach_layers(vector<Layer*> l) { layers = l; }
    virtual void update(vector<Tensor<double>> gradients) = 0;
};

class SGD : public Optimizer
{
  private:
    double decay;

  public:
    SGD(double alpha = 1e-3, double decay = 1)
      : Optimizer(alpha)
      , decay(decay)
    {
    }
    void update(vector<Tensor<double>> gradients);
};

class Adam : public Optimizer
{
    private:
      double beta1, beta2;
      vector<Tensor<double>> updates1;
      vector<Tensor<double>> updates2;
      double t;

    public:
      Adam(double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999);
      void attach_layers(vector<Layer*> l);
      void update(vector<Tensor<double>> gradients);
};


#endif // __OPTIMIZER_HPP__
