
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

    std::vector<double> result = {};

    for (size_t i = 0; i < y_true.size(); i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

Loss mae() {
    return Loss(&mae_f, &mae_d);
}