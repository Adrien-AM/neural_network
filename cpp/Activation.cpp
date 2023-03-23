#include "Activation.hpp"

#include "Utils.hpp"

// -- Linear --

vector<double>
Linear::compute(vector<double> x) const
{
    return x;
}

vector<vector<double>>
Linear::derivative(vector<double> x) const
{
    vector<vector<double>> result(x.size());
    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = vector<double>(x.size());
        result[i][i] = 1;
    }
    return result;
}

// -- ReLU --

vector<double>
ReLU::compute(vector<double> x) const
{
    vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = x[i] > 0 ? x[i] : 0;
    }
    return result;
}

vector<vector<double>>
ReLU::derivative(vector<double> x) const
{
    vector<vector<double>> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = vector<double>(x.size());
        result[i][i] = x[i] > 0 ? 1 : 0;
    }
    return result;
}

// -- Sigmoid --

vector<double>
Sigmoid::compute(vector<double> x) const
{
    vector<double> result(x.size());
    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = 1 / (1 + exp(-x[i]));
    }
    return result;
}

vector<vector<double>>
Sigmoid::derivative(vector<double> x) const
{
    vector<vector<double>> result(x.size());
    vector<double> sigmas = this->compute(x);
    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i = 0; i < x.size(); i++) {
        result[i] = vector<double>(x.size());
        result[i][i] = sigmas[i] * (1 - sigmas[i]);
    }
    return result;
}

// -- Softmax --

vector<double>
Softmax::compute(vector<double> x) const
{
    double sum = 0;
    vector<double> result(x.size());
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

vector<vector<double>>
Softmax::derivative(vector<double> x) const
{
    unsigned int size = x.size();
    vector<vector<double>> result(size);

#ifdef DEBUG
    printf("Actv values :\n");
    print_vector(x);
#endif

    for (unsigned int i = 0; i < size; i++) {
        result[i] = vector<double>(size);
        for (unsigned int j = 0; j <= i; j++) {
            // note : this is symetric !
            result[i][j] = x[i] * ((i == j) - x[j]);
            result[j][i] = result[i][j];

#ifdef DEBUG
// printf("(%u,%u) x %f, j %f, result %f\n", i, j, x[i], x[j], result[i]);
#endif
        }
    }

    return result;
}