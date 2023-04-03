
#include <cmath>
#include <iostream>


#include "Loss.hpp"
#include "Utils.hpp"

// Mean Absolute Error evaluation
double
mae_f(Tensor<double> y_true, Tensor<double> y_pred)
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

// Mean Absolute Error derivative
Tensor<double>
mae_d(Tensor<double> y_true, Tensor<double> y_pred)
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

Loss
MeanAbsoluteError()
{
    return Loss(&mae_f, &mae_d);
}

// Mean Squared Error evaluation
double
mse_f(Tensor<double> y_true, Tensor<double> y_pred)
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
mse_d(Tensor<double> y_true, Tensor<double> y_pred)
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

Loss
MeanSquaredError()
{
    return Loss(&mse_f, &mse_d);
}

// Cross Entropy evaluation
double
ce_f(Tensor<double> y_true, Tensor<double> y_pred)
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
ce_d(Tensor<double> y_true, Tensor<double> y_pred)
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

Loss
CategoricalCrossEntropy()
{
    return Loss(&ce_f, &ce_d);
}