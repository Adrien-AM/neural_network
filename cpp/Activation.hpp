#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <vector>

class Activation
{
  public:
    virtual double compute(double) const = 0;
    virtual double derivative(double) const = 0;
};

class Linear : public Activation
{
  public:
    double compute(double) const;
    double derivative(double) const;
};

class ReLU : public Activation
{
  public:
    double compute(double) const;
    double derivative(double) const;
};

class Sigmoid : public Activation
{
  public:
    double compute(double) const;
    double derivative(double) const;
};

class Softmax : public Activation
{
  private:
    unsigned int nb_of_classes;

  public:
    Softmax(unsigned int nb_of_classes);
    double compute(double) const;
    double derivative(double) const;
};

#endif // __ACTIVATION_H__