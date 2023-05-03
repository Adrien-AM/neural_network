#include <cmath>
#include <iostream>

#include "Loss.hpp"
#include "Utils.hpp"

// AUTODIFF
Tensor<double>
Loss::derivate(const Tensor<double>& real, const Tensor<double>& predicted)
{
    this->graph->reset();
    this->evaluate(real, predicted);
    this->graph->backward();
    Tensor<double> result(predicted.shape());

    Variable<double>** input_data = inputs.data();
    double* result_data = result.data();
    for (size_t i = 0; i < result.total_size(); i++) {
        result_data[i] = input_data[i]->gradient;
    }

    return result;
}

void
Loss::create_variables(const Tensor<double>& predicted)
{
    size_t size = predicted.total_size();
    Tensor<double> copy = predicted;
    double* data = copy.data();
    inputs = predicted.shape();
    if (size == 0)
        return;
    Variable<double>** variables = inputs.data();
    for (size_t i = 0; i < size; i++) {
        variables[i] = new Variable<double>(data[i]);
    }
}

Loss::~Loss()
{
    delete graph;
}

double
MeanAbsoluteError::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    size_t size = y_true.size();
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += std::fabs(error);
    }
    return total / size;
}

Tensor<double>
MeanAbsoluteError::derivate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    size_t size = y_true.size();
    if (0 == size)
        return {};

    Tensor<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

// Mean Squared Error evaluation
double
MeanSquaredError::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    size_t size = y_true.size();
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += error * error;
    }

    return total / (2 * size);
}

// Mean Squared Error derivative
Tensor<double>
MeanSquaredError::derivate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    size_t size = y_true.size();
    if (0 == size)
        return {};

    Tensor<double> result = Tensor<double>(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

// Cross Entropy evaluation
double
CategoricalCrossEntropy::evaluate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    double loss = 0;
    size_t size = y_true.size();
    for (size_t i = 0; i < size; i++) {
        // y_pred cannot be 0 after a softmax because of exp
        loss -= (y_true[i] * log(y_pred[i]));
    }

    return loss;
}

// Cross Entropy derivative
Tensor<double>
CategoricalCrossEntropy::derivate(const Tensor<double>& y_true, const Tensor<double>& y_pred)
{
    size_t size = y_true.size();
    Tensor<double> result = Tensor<double>(size);

#ifdef DEBUG
    printf("Derivatives predictions :\n");
    print_vector(y_pred);
#endif
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; i++) {
        result[i] = -(y_true[i] / y_pred[i]);
    }

    return result;
}

double
SSIM::evaluate(const Tensor<double>& real, const Tensor<double>& predicted)
{
    // Assume 1 color channel for grayscale
    Tensor<double> img1 = real.at(0);
    Tensor<double> img2 = predicted.at(0);

    create_variables(img2);
    Variable<double>* k1 = new Variable<double>(0.01);
    Variable<double>* k2 = new Variable<double>(0.03);

    size_t height = img1.shape()[0];
    size_t width = img1.shape()[1];

    // Compute means and variances
    Operation<double>* mean1 = new Variable<double>(0.0);
    Operation<double>* mean2 = new Variable<double>(0.0);

    Operation<double>* var1 = new Variable<double>(0.0);
    Operation<double>* var2 = new Variable<double>(0.0);
    Operation<double>* cov = new Variable<double>(0.0);
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            mean1 = new Add<double>(mean1, new Variable<double>(img1.at(i)[j]));
            mean2 = new Add<double>(mean2, inputs.at(i)[j]);
        }
    }
    Variable<double>* s = new Variable<double>(height * width);
    mean1 = new Div<double>(mean1, s);
    mean2 = new Div<double>(mean2, s);

    for (size_t j = 0; j < width; j++) {
        for (size_t i = 0; i < height; i++) {
            Operation<double>* diff1 = new Sub<double>(new Variable<double>(img1.at(i)[j]), mean1);
            Operation<double>* diff2 = new Sub<double>(inputs.at(i)[j], mean2);
            var1 = new Add<double>(var1, new Mul<double>(diff1, diff1));
            var2 = new Add<double>(var2, new Mul<double>(diff2, diff2));
            cov = new Add<double>(cov, new Mul<double>(diff1, diff2));
        }
    }

    Variable<double>* n = new Variable<double>(height * width - 1);
    var1 = new Div<double>(var1, n);
    var2 = new Div<double>(var2, n);
    cov = new Div<double>(cov, n);

    // Compute SSIM
    Operation<double>* numerator = new Mul<double>(
      new Add<double>(new Mul<double>(new Variable<double>(2.0), new Mul<double>(mean1, mean2)),
                      k1),
      new Add<double>(new Mul<double>(new Variable<double>(2.0), cov), k2));

    Operation<double>* denominator =
      new Mul<double>(new Add<double>(new Mul<double>(mean1, mean1),
                                      new Add<double>(new Mul<double>(mean2, mean2), k1)),
                      new Add<double>(new Add<double>(var1, var2), k2));

    Operation<double>* ssim = new Div<double>(numerator, denominator);

    // - to make it a Cost function
    ssim = new Sub<double>(new Variable<double>(0.0), ssim);

    if (this->graph != nullptr)
        delete this->graph;
    this->graph = new CompGraph<double>(ssim);
    double x = this->graph->forward();
    return x;
}
