#include "Input.hpp"

Input::Input(std::vector<double> values) : Layer(values.size(), Linear(), false) {
    this->actv_values = values;
}
void Input::forward(const std::vector<double>& inputs) {
    for (unsigned int i = 0; i < inputs.size(); i++) {
        this->values[i] = this->actv_values[i] = inputs[i];
    }
}
void
Input::backprop(Layer*, double, double) {}

void
Input::init(unsigned int) {}

void
Input::summarize() {
    printf("Input | Size %zu\n", this->values.size());
}