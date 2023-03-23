#ifndef __METRIC_HPP__
#define __METRIC_HPP__

#include <algorithm>
#include <vector>

using namespace std;

class Metric
{
  public:
    virtual void add_entry(vector<double> truth, vector<double> output) = 0;
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
    void add_entry(vector<double>, vector<double>);
    double get_result();
};

#endif // __METRIC_HPP__