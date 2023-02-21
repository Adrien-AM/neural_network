
#include <vector>
#include <iostream>
#include <cmath>

#include "Loss.hpp"

// Mean Absolute Error evaluation
double
mae_f(std::vector<double> y_true, std::vector<double> y_pred)
{       
    if (0 == y_true.size())
        return 0;
    double total = 0;
    for (int i = 0; i < y_true.size(); i++) {
        double error = y_pred[i] - y_true[i];
        total += std::fabs(error);
    }
    return total / y_true.size();
}

// Mean Absolute Error derivative
std::vector<double>
mae_d(std::vector<double> y_true, std::vector<double> y_pred)
{
    if (0 == y_true.size())
        return {};

    std::vector<double> result(y_true.size());

    for (size_t i = 0; i < y_true.size(); i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

Loss mae() {
    return Loss(&mae_f, &mae_d);
}

// Mean Squared Error evaluation
double
mse_f(std::vector<double> y_true, std::vector<double> y_pred)
{
    if (0 == y_true.size())
        return 0;
    double total = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double error = y_pred[i] - y_true[i];
        total += error * error;
    }

    return total / y_true.size();
}

// Mean Squared Error derivative
std::vector<double>
mse_d(std::vector<double> y_true, std::vector<double> y_pred)
{
    if (0 == y_true.size())
        return {};

    std::vector<double> result = std::vector<double>(y_true.size());
    for (size_t i = 0; i < y_true.size(); i++) {
        result[i] = 2 * (y_pred[i] - y_true[i]);
    }

    return result;
}

Loss mse()
{
    return Loss(&mse_f, &mse_d);
}