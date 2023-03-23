#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class Activation
{
  public:
    virtual vector<double> compute(vector<double>) const = 0;
    virtual vector<vector<double>> derivative(vector<double>) const = 0;
};

class Linear : public Activation
{
  public:
    vector<double> compute(vector<double>) const;
    vector<vector<double>> derivative(vector<double>) const;
};

class ReLU : public Activation
{
  public:
    vector<double> compute(vector<double>) const;
    vector<vector<double>> derivative(vector<double>) const;
};

class Sigmoid : public Activation
{
  public:
    vector<double> compute(vector<double>) const;
    vector<vector<double>> derivative(vector<double>) const;
};

class Softmax : public Activation
{
  public:
    vector<double> compute(vector<double>) const;
    vector<vector<double>> derivative(vector<double>) const;
};

#endif // __ACTIVATION_H__