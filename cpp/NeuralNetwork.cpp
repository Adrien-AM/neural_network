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
        this->layers[i]->init(this->layers[i - 1]->output_shape());
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
    Tensor<double> y = inputs;
    for (Layer*& layer : this->layers) {
        if (y.shape().size() == 1)
            y.add_dimension();
        y = layer->forward(y);
    }

    return y;
}

void
NeuralNetwork::fit(Dataset<double>& data, size_t batch_size, size_t epochs)
{
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        printf("- Epoch %zu/%zu -- ", epoch + 1, epochs);
        fflush(stdout);
        double epoch_loss = 0;

        size_t data_size = data.size();

        for (size_t i = 0; i < data_size; i += batch_size)
        {
            vector<Tensor<double>> batch = data.get_item(i, batch_size);
            Tensor<double>& input = batch[0];
            Tensor<double>& output = batch[1];

            Tensor<double> predicted = this->predict(input);
            double curr_loss = this->loss.evaluate(output, predicted);
            printf("Current loss : %f\n", curr_loss);
            epoch_loss += curr_loss;
            #ifdef DEBUG
            if (epoch_loss != epoch_loss) {
                printf("Network diverged during training.\n");
                exit(0);
            }
            #endif
            this->loss.backward();
            optimizer.update();
        }
        epoch_loss /= (data_size / batch_size);
        printf("Epoch %zu : loss = %f\n", epoch, epoch_loss);
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
        Tensor<double> prediction = this->predict(inputs[i]);
        metric->add_entry(outputs[i], prediction);
        double current_loss = loss.evaluate(outputs[i], prediction);
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