#include "Metric.hpp"

#include "Utils.hpp"

void
Accuracy::add_entry(Tensor<double> truth, Tensor<double> output)
{
    size_t predicted_class = 0;
    for (size_t i = 0; i < output.size(); i++) 
      if (output[i] > output[predicted_class])
          predicted_class = i;

    size_t real_class = 0;
    for (size_t i = 0; i < truth.size(); i++)
      if (truth[i] > truth[real_class])
          real_class = i;
    if (predicted_class == real_class)
        this->positive += 1;
    this->total += 1;
}

double
Accuracy::get_result()
{
    return (double)this->positive / this->total;
}