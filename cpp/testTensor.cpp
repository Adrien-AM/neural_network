#include "Tensor.hpp"

#include <iostream>

using namespace std;

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

    vector<Tensor<double>> curr(x.shape().size() - 1);
    for (size_t dim = 0; dim < curr.size(); dim++) {
        curr[dim] = (dim == 0 ? x : curr[dim - 1]).at(0);
    }

    vector<size_t> indices(x.shape().size());
    for (size_t i = 0; i < size; i++) {
        size_t dim = indices.size() - 1;
        while (dim != 0 && indices[dim] == x.shape()[dim]) {
            indices[dim] = 0;
            dim--;
            indices[dim]++;
        }
        for (size_t d = dim; d < curr.size(); d++) {
            curr[d] = (d == 0 ? x : curr[d - 1]).at(indices[d]);
        }
        out[i] = curr.back()[indices.back()];
        indices.back()++;
    }

    out.print();
}
