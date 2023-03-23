
#include <cmath>
#include <iostream>
#include <vector>

#include "Loss.hpp"
#include "Utils.hpp"

// Mean Absolute Error evaluation
double
mae_f(vector<double> y_true, vector<double> y_pred)
{
    unsigned int size = y_true.size();
    if (0 == size)
        return 0;
    double total = 0;
    for (unsigned int i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += std::fabs(error);
    }
    return total / size;
}

// Mean Absolute Error derivative
vector<double>
mae_d(vector<double> y_true, vector<double> y_pred)
{
    unsigned int size = y_true.size();
    if (0 == size)
        return {};

    vector<double> result(size);

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
mse_f(vector<double> y_true, vector<double> y_pred)
{
    unsigned int size = y_true.size();
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
vector<double>
mse_d(vector<double> y_true, vector<double> y_pred)
{
    unsigned int size = y_true.size();
    if (0 == size)
        return {};

    vector<double> result = vector<double>(size);
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
ce_f(vector<double> y_true, vector<double> y_pred)
{
    double loss = 0;
    unsigned int size = y_true.size();
    for (size_t i = 0; i < size; i++) {
        // y_pred cannot be 0 after a softmax because of exp
        loss -= (y_true[i] * log(y_pred[i]));
    }

    return loss;
}

// Cross Entropy derivative
vector<double>
ce_d(vector<double> y_true, vector<double> y_pred)
{
    unsigned int size = y_true.size();
    vector<double> result = vector<double>(size);

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