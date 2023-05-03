#include "Tensor.hpp"

#include <iostream>

using namespace std;

Tensor<double> test() {
    return Tensor<double>(vector<double>{ 1, 2 });
}

int
main(void)
{
    Tensor<double> x = vector<size_t>({ 2, 3, 2 });
    x.at(0).at(0) = { 1, 2 };
    x.at(0).at(1) = { 3, 4 };
    x.at(0).at(2) = { 5, 6 };
    x.at(1).at(0) = { 7, 8 };
    x.at(1).at(1) = { 9, 10 };
    x.at(1).at(2) = { 11, 12 };

    size_t size = 1;

    for (int dim = x.shape().size() - 1; dim >= 0; dim--) {
        size *= x.shape()[dim];
    }
    Tensor<double> out = vector<double>(size);

    memcpy(out.data(), x.data(), size * sizeof(double));
    out.print();

    Tensor<double> y = vector<size_t>({ 2, 3, 2 });

    memcpy(y.data(), x.data(), size * sizeof(double));
    y.print();

    y.add_dimension();
    y.print();
    y.print_shape();

    Tensor<double> tmp = test();
    x.at(0) = tmp;
    x.at(0).print();
}
