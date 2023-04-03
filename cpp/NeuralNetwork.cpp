#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(vector<size_t> input_shape, vector<Layer*> layers, Loss loss)
  : input_shape(input_shape)
  , layers(layers)
  , loss(loss)
{
    if (this->layers.size() == 0)
        return;

    this->layers[0]->init(input_shape);
    for (size_t i = 1; i < this->layers.size(); i++) {
        this->layers[i]->init(this->layers[i - 1]->output_values.shape());
    }
}

Tensor<double>
NeuralNetwork::predict(const Tensor<double>& inputs)
{
    Tensor<double> moving_inputs = inputs;
    for (Layer*& layer : this->layers) {
        layer->forward(moving_inputs);
        moving_inputs = layer->output_values;
    }

    // Last element actv values are the output
    return moving_inputs;
}

void
NeuralNetwork::backpropagation(const Tensor<double>& real, const Tensor<double>& inputs)
{
    Tensor<double> partial_errors =
      this->loss.derivate(real, this->layers.back()->output_values);
    this->layers.back()->errors = partial_errors;
#ifdef DEBUG
    printf("Partial errors (loss derivatives) : ");
    print_vector(partial_errors);
#endif

    for (size_t i = this->layers.size() - 1; i > 0; i--) {
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
NeuralNetwork::fit(const Tensor<double>& inputs,
                   const Tensor<double>& outputs,
                   double learning_rate,
                   double momentum,
                   size_t batch_size,
                   size_t epochs)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%zu) and outputs (%zu) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    this->alpha = learning_rate;
    this->gamma = momentum;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        printf("- Epoch %zu -- ", epoch + 1);
        fflush(stdout);
        double loss = 0;
        vector<size_t> data_idx(inputs.size());
        std::iota(data_idx.begin(), data_idx.end(), 0);
        vector<size_t> sample_idx(batch_size);
        std::sample(data_idx.begin(),
                    data_idx.end(),
                    sample_idx.begin(),
                    batch_size,
                    std::mt19937(std::random_device()()));

        for (size_t row = 0; row < batch_size; row++) {
            const Tensor<double>& input = inputs.at(sample_idx[row]);
            const Tensor<double> output = outputs.at(sample_idx[row]);
            reset_values();
            Tensor<double> predicted = this->predict(input);
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
        // this->alpha *= 0.9;
        printf("Mean loss : %f\n", loss);
    }
}

double
NeuralNetwork::evaluate(const Tensor<double>& inputs,
                        const Tensor<double>& outputs,
                        Loss loss,
                        Metric* metric)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%zu) and outputs (%zu) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    double total_loss = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        Tensor<double> prediction = this->predict(inputs.at(i));
        metric->add_entry(outputs.at(i), prediction);
        double current_loss = loss.evaluate(outputs.at(i), prediction);
        total_loss += (current_loss - total_loss) / (i + 1); // moving average
    }

    return total_loss;
}

void
NeuralNetwork::summarize()
{
    printf("\nNeural Net :\n");
    for (size_t i = 0; i < this->layers.size(); i++) {
        printf("--Layer %zu--\n\t", i);
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