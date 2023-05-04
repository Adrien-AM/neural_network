#include "Activation.hpp"

#include "Utils.hpp"

// -- Linear --

Tensor<double>
Linear::compute(Tensor<double> x) const
{
    return x;
}

Tensor<double>
Linear::derivative(Tensor<double> x) const
{
    Tensor<double> result(vector<size_t>({ x.size(), x.size() }));
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (size_t i = 0; i < x.size(); i++) {
        result.at(i)[i] = 1;
    }
    return result;
}

// -- ReLU --

Tensor<double>
ReLU::compute(Tensor<double> x) const
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.size(); i++) {
        double xi = x[i];
        result[i] = xi > 0 ? xi : 0;
    }
    return result;
}

Tensor<double>
ReLU::derivative(Tensor<double> x) const
{
    Tensor<double> result(vector<size_t>({ x.size(), x.size() }));
    for (size_t i = 0; i < x.size(); i++) {
        result.at(i)[i] = x[i] > 0 ? 1 : 0;
    }
    return result;
}

// -- Sigmoid --

Tensor<double>
Sigmoid::compute(Tensor<double> x) const
{
    Tensor<double> result(x.shape());
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = 1 / (1 + exp(-x[i]));
    }
    return result;
}

Tensor<double>
Sigmoid::derivative(Tensor<double> x) const
{
    Tensor<double> result(vector<size_t>({ x.size(), x.size() }));
    Tensor<double> sigmas = this->compute(x);
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (size_t i = 0; i < x.size(); i++) {
        double si = sigmas[i];
        result.at(i)[i] = si * (1 - si);
    }
    return result;
}

// -- Softmax --

Tensor<double>
Softmax::compute(Tensor<double> x) const
{
    double sum = 0;
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.size(); i++) {
        double e = exp(x[i]);
        sum += e;
#ifdef DEBUG
        if (sum != sum) {
            printf("Diverged : %f.\n", e);
            print_vector(x);
            exit(0);
        }
#endif
        result[i] = e;
    }

    for (size_t i = 0; i < x.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

Tensor<double>
Softmax::derivative(Tensor<double> x) const
{
    size_t size = x.size();
    Tensor<double> result(vector<size_t>({ size, size }));

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j <= i; j++) {
            double& rij = result.at(i)[j];
            // note : this is symetric !
            rij = x[i] * ((i == j) - x[j]);
            result.at(j)[i] = rij;
        }
    }

    return result;
}