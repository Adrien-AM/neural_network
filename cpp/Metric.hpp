#ifndef __METRIC_HPP__
#define __METRIC_HPP__

#include <algorithm>

#include "Tensor.hpp"

using namespace std;

class Metric
{
  public:
    virtual void add_entry(Tensor<double> truth, Tensor<double> output) = 0;
    virtual double get_result() = 0;
};

class Accuracy : public Metric
{
  private:
    size_t positive;
    size_t total;

  public:
    Accuracy()
      : positive(0)
      , total(0){}
    void add_entry(Tensor<double>, Tensor<double>);
    double get_result();
};

#endif // __METRIC_HPP__