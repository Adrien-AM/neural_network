#include "Utils.hpp"
#include <iostream>
#include <vector>

using namespace std;

int
main()
{
    Tensor<double> img = vector<size_t>({ 2, 4, 4 });
    Tensor<double> c1 = img.at(0);
    c1.at(0) = { 1, 1, 1, 1 };
    c1.at(1) = { 1, 1, 1, 1 };
    c1.at(2) = { 1, 1, 1, 1 };
    c1.at(3) = { 1, 1, 1, 1 };

    Tensor<double> c2 = img.at(1);
    c2.at(0) = { 1, 1, 1, 1 };
    c2.at(1) = { 1, 1, 1, 1 };
    c2.at(2) = { 1, 1, 1, 1 };
    c2.at(3) = { 1, 1, 1, 1 };

    Tensor<double> filter = vector<size_t>({ 2, 3, 3 });
    Tensor<double> f1 = filter.at(0);
    f1.at(0) = { 1, 1, 1 };
    f1.at(1) = { 1, 1, 1 };
    f1.at(2) = { 1, 1, 1 };

    Tensor<double> f2 = filter.at(1);
    f2.at(0) = { 1, 1, 1 };
    f2.at(1) = { 1, 1, 1 };
    f2.at(2) = { 1, 1, 1 };

    Tensor<double> result = convolution_product(img, filter, 1);
    result.print();
}
