#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include <vector>

using namespace std;

class Loss
{
  public:
    double (*evaluate)(vector<double> real, vector<double> predicted);
    vector<double> (*derivate)(vector<double> real, vector<double> predicted);

    Loss(double (*evaluate)(vector<double>, vector<double>),
         vector<double> (*derivate)(vector<double>, vector<double>))
      : evaluate(evaluate)
      , derivate(derivate){};
};

Loss
MeanAbsoluteError();

Loss
MeanSquaredError();

Loss
CategoricalCrossEntropy();

#endif // __LOSS_HPP__