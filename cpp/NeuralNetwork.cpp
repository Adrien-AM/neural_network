#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(unsigned int input_size, std::vector<Layer*> layers, Loss l)
  : input_size(input_size)
  , layers(layers)
  , loss(l)
{
    if (this->layers.size() == 0)
        return;

    this->layers[0]->init(input_size);
    for (unsigned int i = 1; i < this->layers.size(); i++) {
        this->layers[i]->init(this->layers[i - 1]->size());
    }
}

void
NeuralNetwork::reset_values()
{
    for (Layer*& l : this->layers) {
        std::fill(l->actv_values.begin(), l->actv_values.end(), 0);
        std::fill(l->values.begin(), l->values.end(), 0);
    }
}

void
NeuralNetwork::reset_errors()
{
    for (Layer*& l : this->layers) {
        std::fill(l->delta_errors.begin(), l->delta_errors.end(), 0);
        std::fill(l->errors.begin(), l->errors.end(), 0);
    }
}

std::vector<double>
NeuralNetwork::feed_forward(std::vector<double> inputs)
{
    for (Layer*& layer : this->layers) {
        layer->forward(inputs);
        inputs = layer->actv_values;
    }

    // Last element actv values are the output
    return this->layers.back()->actv_values;
}

void
NeuralNetwork::backpropagation(std::vector<double> real, std::vector<double> inputs)
{
    std::vector<double> partial_errors = this->loss.derivate(real, this->layers.back()->actv_values);
    this->layers.back()->errors = partial_errors;

    for (unsigned int i = this->layers.size() - 1; i > 0; i--) {
        this->layers[i]->backprop(this->layers[i - 1], this->alpha);
    }

    Input fake_input(inputs);
    this->layers[0]->backprop(&fake_input, this->alpha);

    return;
}

void
NeuralNetwork::fit(std::vector<std::vector<double>> inputs,
                   std::vector<std::vector<double>> outputs,
                   double learning_rate,
                   unsigned int epochs)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%u) and outputs (%u) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    this->alpha = learning_rate;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
        printf("- Epoch %u -- ", epoch + 1);
        double loss = 0;
        for (unsigned int row = 0; row < inputs.size(); row++) {
            reset_values();
            std::vector<double> predicted = this->predict(inputs[row]);
            loss += this->loss.evaluate(outputs[row], predicted);
            reset_errors();
            this->backpropagation(outputs[row], inputs[row]);
        }
        printf("Mean loss : %f\n", loss / inputs.size());
    }
}

std::vector<double>
NeuralNetwork::predict(std::vector<double> inputs)
{
    this->reset_values();

    return this->feed_forward(inputs);
}

void NeuralNetwork::summarize()
{
    printf("\nNeural Net :\n");
    for (unsigned int i = 0; i < this->layers.size(); i++) {
        printf("--Layer %u--\n\t", i);
        this->layers[i]->summarize();
        printf("------------\n");
    }
    printf("\n");
}

NeuralNetwork::~NeuralNetwork()
{
    for (Layer*& l : this->layers) {
        delete l;
    }
}