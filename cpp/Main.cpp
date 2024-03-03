#include <iostream>
#include <random>

#include "Activation.hpp"
#include "Dataset.hpp"
#include "Dense.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"
#include "Optimizer.hpp"

using namespace std;

double
f(double x)
{
    return 0.4 * x - 2;
}

class CustomData : public Dataset<double>
{
  private:
    mt19937 g;
    uniform_real_distribution<> d;
    size_t n;

  public:
    CustomData(double nmin = -10, double nmax = 10, double n = 500)
      : Dataset<double>()
    {
        random_device rd;
        mt19937 gen(rd());
        this->g = gen;
        uniform_real_distribution<> dis(nmin, nmax);
        this->d = dis;
        this->n = n;
    }

    vector<Tensor<double>> get_item(size_t index, size_t nb)
    {
        (void)index;
        vector<Tensor<double>> result(2);
        Tensor<double> x = vector<size_t>({ nb, 1 });
        Tensor<double> y = vector<size_t>({ nb, 1 });
        for (size_t k = 0; k < nb; k++) {
            x({ k, 0 }) = this->d(this->g);
            y({ k, 0 }) = f(x({ k, 0 }));
        }
        result[0] = x;
        result[1] = y;
        return result;
    }

    size_t size() const { return this->n; }
};

int
main()
{
    vector<Layer*> layers = { new Dense(1, true), new Dense(1, true) };
    MeanSquaredError loss;
    SGD optimizer(1e-3);
    NeuralNetwork nn({ 1 }, layers, loss, optimizer);
    CustomData data;

    nn.fit(data, 1, 5);

    vector<Tensor<double>> validation = data.get_item(0, 1);
    printf("%f should be %f\n", (double)nn.predict(validation[0])({0, 0}), (double)validation[1]({0, 0}));

    // double newrand = dis(gen);
    // Tensor<double> input(vector<size_t>{ 1, 1 });
    // input({ 0, 0 }) = newrand;
    // printf("f(%f) -> %f (should be %f)\n", newrand, nn.predict(input)({ 0, 0 }), f(newrand));

    return 0;
}