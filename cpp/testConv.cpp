#include "Utils.hpp"
#include <iostream>
#include <vector>

using namespace std;

int
main()
{
    vector<double> img = { 1, 0, 2, 1, 
    1, 2, 3, 1,
     2, 1, 2, 0,
      1, 3, 2, 1 };
    vector<double> filter = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    vector<double> result = convolution_product(img, filter, 4, 1);
    print_vector(result);
    return 0;
}
