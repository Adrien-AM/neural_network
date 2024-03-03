// #include "Input.hpp"

// Input::Input(Tensor<double> values)
// {
//     this->output_values = values;
// }

// void
// Input::init(vector<size_t>)
// {
// }

// Tensor<double>
// Input::forward(const Tensor<double>& inputs)
// {
//     this->output_values = inputs;
// }

// void
// Input::summarize() const
// {
//     printf("Input | Size %zu\n", this->output_values.size());
// }

// size_t
// Input::size() const
// {
//     return this->output_values.total_size();
// }

// void
// Input::print_layer() const
// {
//     printf("Input values :\n");
//     this->output_values.print();
// }

// Input*
// Input::clone() const
// {
//     return new Input(this->output_values);
// }