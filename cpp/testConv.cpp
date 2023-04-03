#include "Utils.hpp"
#include <iostream>
#include <vector>

using namespace std;

int
main()
{
    Tensor<double> img = vector<size_t>({ 4, 4 });
    img.at(0) = { 1, 0, 2, 1 };
    img.at(1) = { 1, 2, 3, 1 };
    img.at(2) = { 2, 1, 2, 0 };
    img.at(3) = { 1, 3, 2, 1 };

    Tensor<double> filter = vector<size_t>({ 3, 3 });
    filter.at(0) = { 1, 1, 1 };
    filter.at(1) = { 1, 1, 1 };
    filter.at(2) = { 1, 1, 1 };
    Tensor<double> result = convolution_product(img, filter, 4, 1);
    result.print();
    return 0;
}
