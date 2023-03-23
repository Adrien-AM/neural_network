#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(unsigned int input_size, std::vector<Layer*> layers, Loss loss)
  : input_size(input_size)
  , layers(layers)
  , loss(loss)
{
    if (this->layers.size() == 0)
        return;

    this->layers[0]->init(input_size);
    for (unsigned int i = 1; i < this->layers.size(); i++) {
        this->layers[i]->init(this->layers[i - 1]->size());
    }
}

std::vector<double>
NeuralNetwork::predict(const std::vector<double>& inputs)
{
    std::vector<double> moving_inputs = inputs;
    for (Layer*& layer : this->layers) {
        layer->forward(moving_inputs);
        moving_inputs = layer->output_values;
    }

    // Last element actv values are the output
    return moving_inputs;
}

void
NeuralNetwork::backpropagation(const std::vector<double>& real, const std::vector<double>& inputs)
{
    std::vector<double> partial_errors =
      this->loss.derivate(real, this->layers.back()->output_values);
    this->layers.back()->errors = partial_errors;
#ifdef DEBUG
    printf("Partial errors (loss derivatives) : ");
    print_vector(partial_errors);
#endif

    for (unsigned int i = this->layers.size() - 1; i > 0; i--) {
        this->layers[i]->backprop(this->layers[i - 1], this->alpha, this->gamma);
    }

    Input fake_input(inputs);
    this->layers[0]->backprop(&fake_input, this->alpha, this->gamma);

    return;
}

void
NeuralNetwork::reset_values()
{
    for (Layer*& layer : this->layers) {
        layer->reset_values();
    }
}

void
NeuralNetwork::reset_errors()
{
    for (Layer*& layer : this->layers) {
        layer->reset_errors();
    }
}

void
NeuralNetwork::fit(const std::vector<std::vector<double>>& inputs,
                   const std::vector<std::vector<double>>& outputs,
                   double learning_rate,
                   double momentum,
                   size_t batch_size,
                   unsigned int epochs)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%zu) and outputs (%zu) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    this->alpha = learning_rate;
    this->gamma = momentum;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
        printf("- Epoch %u -- ", epoch + 1);
        fflush(stdout);
        std::vector<size_t> data_idx(inputs.size());
        std::iota(data_idx.begin(), data_idx.end(), 0);
        std::vector<size_t> sample_idx(batch_size);
        std::sample(data_idx.begin(),
                    data_idx.end(),
                    sample_idx.begin(),
                    batch_size,
                    std::mt19937(std::random_device()()));

        double loss = 0;
        for (unsigned int row = 0; row < batch_size; row++) {
            const std::vector<double>& input = inputs[sample_idx[row]];
            const std::vector<double> output = outputs[sample_idx[row]];
            this->reset_values();
            std::vector<double> predicted = this->predict(input);
            double curr_loss = this->loss.evaluate(output, predicted);

            loss += (curr_loss - loss) / (row + 1); // Moving average
            if (loss != loss) {
                printf("Networked diverged during training.\n");
                exit(0);
            }
            // print_vector(predicted);
            // print_vector(outputs[row]);
            // printf("Curr loss : %f - New loss : %f\n--\n", curr_loss, loss);
            this->reset_errors();
            this->backpropagation(output, input);
        }
        this->alpha *= 0.95;
        printf("Mean loss : %f\n", loss);
    }
}

double
NeuralNetwork::evaluate(const std::vector<std::vector<double>>& inputs,
                        const std::vector<std::vector<double>>& outputs,
                        Loss loss)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%zu) and outputs (%zu) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    double total_loss = 0;
    for (unsigned int i = 0; i < inputs.size(); i++) {
        double current_loss = loss.evaluate(outputs[i], this->predict(inputs[i]));
        total_loss += (current_loss - total_loss) / (i + 1); // moving average
    }

    return total_loss;
}

void
NeuralNetwork::summarize()
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