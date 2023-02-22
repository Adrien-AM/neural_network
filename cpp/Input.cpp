#include "Input.hpp"

Input::Input(std::vector<double> values) : Layer(Linear(), values.size(), false) {
    this->actv_values = values;
}
void Input::forward(std::vector<double> inputs) {
    for (unsigned int i = 0; i < inputs.size(); i++) {
        this->values[i] = this->actv_values[i] = inputs[i];
    }
}
void
Input::backprop(Layer*, double) {}

void
Input::init(unsigned int) {}

void
Input::summarize() {
    printf("Input | Size %u\n", this->values.size());
}