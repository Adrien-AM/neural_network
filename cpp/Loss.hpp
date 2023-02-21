#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include <vector>

class Loss
{
  public:
    double (*evaluate)(std::vector<double> real, std::vector<double> predicted);
    std::vector<double> (*derivate)(std::vector<double> real, std::vector<double> predicted);

    Loss(double (*evaluate)(std::vector<double>, std::vector<double>),
         std::vector<double> (*derivate)(std::vector<double>, std::vector<double>))
      : evaluate(evaluate)
      , derivate(derivate){};
};

Loss
mae();

Loss
mse();

#endif // __LOSS_HPP__