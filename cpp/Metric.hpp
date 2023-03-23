#ifndef __METRIC_HPP__
#define __METRIC_HPP__

#include <algorithm>
#include <vector>

class Metric
{
  public:
    virtual void add_entry(std::vector<double> truth, std::vector<double> output) = 0;
    virtual double get_result() = 0;
};

class Accuracy : public Metric
{
  private:
    unsigned int positive;
    unsigned int total;

  public:
    Accuracy()
      : positive(0)
      , total(0){}
    void add_entry(std::vector<double>, std::vector<double>);
    double get_result();
};

#endif // __METRIC_HPP__