#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(vector<size_t> input_shape,
                             vector<Layer*> layers,
                             Loss& loss,
                             Optimizer& opti)
  : input_shape(input_shape)
  , layers(layers)
  , loss(loss)
  , optimizer(opti)
{
    if (this->layers.size() == 0)
        return;

    this->layers[0]->init(input_shape);
    for (size_t i = 1; i < this->layers.size(); i++) {
        this->layers[i]->init(this->layers[i - 1]->output_values.shape());
    }
    optimizer.attach_layers(this->layers);
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
  : input_shape(other.input_shape)
  , loss(other.loss)
  , optimizer(other.optimizer)
{
    this->layers = vector<Layer*>(other.layers.size());
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i] = other.layers[i]->clone();
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

vector<Tensor<double>>
NeuralNetwork::backpropagation(const Tensor<double>& real, const Tensor<double>& inputs)
{
    Tensor<double> partial_errors = this->loss.derivate(real, this->layers.back()->output_values);
    this->layers.back()->errors = partial_errors;

    vector<Tensor<double>> gradients(layers.size());

    for (size_t i = this->layers.size() - 1; i > 0; i--) {
        gradients[i] = this->layers[i]->backprop(this->layers[i - 1]);
    }

    Input fake_input(inputs);
    gradients[0] = this->layers[0]->backprop(&fake_input);
    return gradients;
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
                   size_t batch_size,
                   size_t epochs)
{
    if (inputs.size() != outputs.size()) {
        fprintf(stderr,
                "Inputs (%zu) and outputs (%zu) should have the same size.\n",
                inputs.size(),
                outputs.size());
    }
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        printf("- Epoch %zu -- ", epoch + 1);
        fflush(stdout);
        double loss = 0;
        vector<Tensor<double>> gradients(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            gradients[i] = Tensor<double>(layers[i]->weights.shape());
        }

        // Sample batch
        vector<size_t> data_idx(inputs.size());
        std::iota(data_idx.begin(), data_idx.end(), 0);
        vector<size_t> sample_idx(batch_size);
        std::sample(data_idx.begin(),
                    data_idx.end(),
                    sample_idx.begin(),
                    batch_size,
                    std::mt19937(std::random_device()()));

#ifdef PARALLEL
#pragma omp parallel for reduction(+ : loss) shared(gradients) num_threads(4)
#endif
        for (size_t row = 0; row < batch_size; row++) {
#ifdef PARALLEL
            NeuralNetwork thread_nn(*this);
            NeuralNetwork* network = &thread_nn;
#else
            NeuralNetwork* network = this;
#endif
            const Tensor<double>& input = inputs.at(sample_idx[row]);
            const Tensor<double>& output = outputs.at(sample_idx[row]);
#ifndef PARALLEL
            network->reset_values();
#endif
            Tensor<double> predicted = network->predict(input);
            double curr_loss = network->loss.evaluate(output, predicted);

            // Atomic not needed because of reduction
            loss += curr_loss;

#ifdef DEBUG
            print_vector(predicted);
            print_vector(outputs.at(row));
            printf("Curr loss : %f - New loss : %f\n--\n", curr_loss, loss);
#endif
            if (loss != loss) {
                printf("Networked diverged during training.\n");
                exit(0);
            }
#ifndef PARALLEL
            network->reset_errors();
#endif
            vector<Tensor<double>> local_grads = network->backpropagation(output, input);
            for (size_t i = 0; i < gradients.size(); i++) {
                double* gradients_data = gradients[i].data();
                double* local_grads_data = local_grads[i].data();
                for (size_t j = 0; j < gradients[i].total_size(); j++) {
#ifdef PARALLEL
#pragma omp atomic
#endif
                    gradients_data[j] += local_grads_data[j];
                }
            }
        }

        for (size_t i = 0; i < gradients.size(); i++) {
            double* gradients_data = gradients[i].data();
            for (size_t j = 0; j < gradients[i].total_size(); j++) {
                gradients_data[j] /= batch_size;
            }
        }
        loss /= batch_size;

        optimizer.update(gradients);
        printf("Mean loss : %f\n", loss);
    }
}

double
NeuralNetwork::evaluate(const Tensor<double>& inputs,
                        const Tensor<double>& outputs,
                        Loss& loss,
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