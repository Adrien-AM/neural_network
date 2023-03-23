#include "Metric.hpp"

#include "Utils.hpp"

void
Accuracy::add_entry(std::vector<double> truth, std::vector<double> output)
{
    unsigned int predicted_class =
      std::distance(output.begin(), std::max_element(output.begin(), output.end())) - 1;
    unsigned int real_class =
      std::distance(truth.begin(), std::max_element(truth.begin(), truth.end())) - 1;
    if (predicted_class == real_class)
        this->positive += 1;
    this->total += 1;
}

double
Accuracy::get_result()
{
    return (double)this->positive / this->total;
}