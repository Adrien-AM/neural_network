#include "Activation.hpp"

#include "Utils.hpp"

// -- Linear --

std::vector<double>
Linear::compute(std::vector<double> x) const
{
    return x;
}

std::vector<std::vector<double>>
Linear::derivative(std::vector<double> x) const
{
    std::vector<std::vector<double>> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = std::vector<double>(x.size());
        result[i][i] = 1;
    }
    return result;
}

// -- ReLU --

std::vector<double>
ReLU::compute(std::vector<double> x) const
{
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = x[i] > 0 ? x[i] : 0;
    }
    return result;
}

std::vector<std::vector<double>>
ReLU::derivative(std::vector<double> x) const
{
    std::vector<std::vector<double>> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = std::vector<double>(x.size());
        result[i][i] = x[i] > 0 ? 1 : 0;
    }
    return result;
}

// -- Sigmoid --

std::vector<double>
Sigmoid::compute(std::vector<double> x) const
{
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = 1 / (1 + exp(-x[i]));
    }
    return result;
}

std::vector<std::vector<double>>
Sigmoid::derivative(std::vector<double> x) const
{
    std::vector<std::vector<double>> result(x.size());
    std::vector<double> sigmas = this->compute(x);
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = std::vector<double>(x.size());
        result[i][i] = sigmas[i] * (1 - sigmas[i]);
    }
    return result;
}

// -- Softmax --

std::vector<double>
Softmax::compute(std::vector<double> x) const
{
    double sum = 0;
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        double e = exp(x[i]);
        sum += e;
        if (sum != sum) {
            printf("Diverged : %f.\n", e);
            print_vector(x);
            exit(0);
        }
        result[i] = e;
    }

    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

std::vector<std::vector<double>>
Softmax::derivative(std::vector<double> x) const
{
    std::vector<std::vector<double>> result(x.size());

#ifdef DEBUG
    printf("Actv values :\n");
    print_vector(x);
#endif

    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = std::vector<double>(x.size());
        for (unsigned int j = 0; j <= i; j++) {
            // note : this is symetric ! maybe optimize later
            result[i][j] = x[i] * ((i == j) - x[j]);
            result[j][i] = result[i][j];

#ifdef DEBUG
// printf("(%u,%u) x %f, j %f, result %f\n", i, j, x[i], x[j], result[i]);
#endif
        }
    }

    return result;
}