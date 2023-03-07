#include "Utils.hpp"

void
print_vector(std::vector<double> vec)
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
print_softmax_output(std::vector<double> vec)
{
    printf("Class : ");
    double max;
    do {
        max = 0;
        int argmax = 0;
        for (unsigned int x = 0; x < vec.size(); x++) {
            if (vec[x] > max) {
                max = vec[x];
                argmax = x;
            }
        }
        printf("%d with %.3f certainty ; ", argmax, max);
        vec[argmax] = 0;
    } while (max > MIN_SOFTMAX_OUTPUT);
    printf("Others too low.\n");
}