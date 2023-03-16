#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <vector>
#include <cmath>
#include <iostream>

class Activation
{
  public:
    virtual std::vector<double> compute(std::vector<double>) const = 0;
    virtual std::vector<std::vector<double>> derivative(std::vector<double>) const = 0;
};

class Linear : public Activation
{
  public:
    std::vector<double> compute(std::vector<double>) const;
    std::vector<std::vector<double>> derivative(std::vector<double>) const;
};

class ReLU : public Activation
{
  public:
    std::vector<double> compute(std::vector<double>) const;
    std::vector<std::vector<double>> derivative(std::vector<double>) const;
};

class Sigmoid : public Activation
{
  public:
    std::vector<double> compute(std::vector<double>) const;
    std::vector<std::vector<double>> derivative(std::vector<double>) const;
};

class Softmax : public Activation
{
  public:
    std::vector<double> compute(std::vector<double>) const;
    std::vector<std::vector<double>> derivative(std::vector<double>) const;
};

#endif // __ACTIVATION_H__