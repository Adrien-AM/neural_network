#include "Utils.hpp"

void
print_vector(vector<size_t> vec)
{
    std::cout << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void
print_softmax_output(Tensor<double> vec)
{
    printf("Class : ");
    double max;
    do {
        max = 0;
        int argmax = 0;
        for (size_t x = 0; x < vec.size(); x++) {
            if (vec[x] > max) {
                max = vec[x];
                argmax = x;
            }
        }
        printf("%d with %.3f certainty ; ", argmax, max);
        vec(argmax) = 0;
    } while (max > MIN_SOFTMAX_OUTPUT);
    printf("Others too low.\n");
}

Tensor<double>
pad_2d(const Tensor<double>& x, size_t size, double value)
{
    (void)size;
    (void)value;
    return x;
}

Tensor<double>
convolution_2d(const Tensor<double>& x, const Tensor<double>& k, const double& bias, size_t stride)
{
    vector<size_t> x_shape = x.shape();
    vector<size_t> k_shape = k.shape();
    size_t result_rows = x_shape[1] - k_shape[1] + 1;
    size_t result_cols = x_shape[2] - k_shape[2] + 1;
    Tensor<double> result(vector<size_t>{ result_rows, result_cols });
    for (size_t i = 0; i < result_rows; i += stride) {
        for (size_t j = 0; j < result_cols; j += stride) {
            double sum = bias;
            for (size_t kc = 0; kc < k_shape[0]; ++kc) {
                for (size_t ki = 0; ki < k_shape[1]; ++ki) {
                    for (size_t kj = 0; kj < k_shape[2]; ++kj) {
                        sum += x({ kc, i + ki, j + kj }) * k({ kc, ki, kj });
                    }
                }
                result({ i, j }) = sum;
            }
        }
    }

    return result;
}