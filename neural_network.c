#include "neural_network.h"
#include "data_utils.h"
#include "layer.h"


struct neural_network
{
    struct layer** layers;
    size_t number_of_layers;
    struct loss loss;
    int use_bias;
};

struct neural_network*
create_model(struct loss loss,
             int use_bias,
             size_t input_size,
             size_t number_of_layers,
             ...)
{
    struct neural_network* nn = (struct neural_network*)malloc(sizeof(struct neural_network));
    nn->number_of_layers = number_of_layers;
    nn->loss = loss;
    nn->use_bias = use_bias;
    nn->layers = (struct layer**)malloc(sizeof(struct layer*) * nn->number_of_layers);
    va_list args;
    va_start(args, number_of_layers);

    for (size_t l = 0; l < number_of_layers; l++) {
        nn->layers[l] = va_arg(args, struct layer*);
        nn->layers[l]->input_size = l == 0 ? input_size : nn->layers[l - 1]->size;
        instanciate_neurons(nn->layers[l]);
    }

    va_end(args);

    return nn;
}

void
free_neural_network(struct neural_network* nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++) {
        struct layer* layer = nn->layers[l];
        for (size_t n = 0; n < layer->size; n++) {
            free(layer->neurons[n]->weights);
            free(layer->neurons[n]);
        }
        free(layer->neurons);
        free(layer);
    }
    free(nn->layers);
    free(nn);
}

void
reset_values(struct neural_network* nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++) {
        for (size_t n = 0; n < nn->layers[l]->size; n++) {
            struct neuron* neuron = nn->layers[l]->neurons[n];
            neuron->value = neuron->bias;
            neuron->actv_value = 0;
        }
    }
}

void
reset_errors(struct neural_network* nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++) {
        for (size_t n = 0; n < nn->layers[l]->size; n++) {
            struct neuron* neuron = nn->layers[l]->neurons[n];
            neuron->error = 0;
            neuron->momentum = 0;
        }
    }
}

void
randomize_weights(struct neural_network* nn, double mu, double sigma)
{
    for (size_t l = 0; l < nn->number_of_layers; l++) {
        for (size_t n = 0; n < nn->layers[l]->size; n++) {
            for (size_t i = 0; i < nn->layers[l]->input_size; i++) {
                nn->layers[l]->neurons[n]->weights[i] = rand_normal(mu, sigma);
            }

            nn->layers[l]->neurons[n]->bias = nn->use_bias ? rand_normal(mu, sigma) : 0;
        }
    }
}

double*
feed_forward(struct neural_network* nn, double inputs[])
{
    reset_values(nn);

    struct layer* layer = nn->layers[0];

    // Create false layer for inputs
    struct layer input_layer;
    struct neuron* neurons[layer->input_size];
    input_layer.neurons = neurons;

    for (size_t i = 0; i < layer->input_size; i++) {
        neurons[i] = malloc(sizeof(struct neuron));
        input_layer.neurons[i]->actv_value = inputs[i];
    }
    layer->forward(layer, &input_layer);
    for (size_t i = 0; i < layer->input_size; i++) {
        free(neurons[i]);
    }
    // End

    for (size_t l = 1; l < nn->number_of_layers; l++) {
        layer = nn->layers[l];
        layer->forward(layer, nn->layers[l - 1]);
    }

    struct layer* last_layer = nn->layers[nn->number_of_layers - 1];
    double* result = (double*)malloc(sizeof(double) * last_layer->size); // last layer values
    for (size_t n = 0; n < last_layer->size; n++) {
        result[n] = last_layer->neurons[n]->actv_value;
    }

    return result;
}

double*
predict(struct neural_network* nn, double* inputs, size_t nb_inputs)
{
    // TODO
    // normalize_inputs(inputs, nn->layers[0]->input_size, nb_inputs);
    (void)nb_inputs;
    return feed_forward(nn, inputs);
}

void
back_propagate(struct neural_network* nn,
               double output[],
               double inputs[],
               double learning_rate,
               double gamma)
{
    reset_errors(nn);

    // Output layer
    struct layer* layer = nn->layers[nn->number_of_layers - 1];

    // Compute loss
    double result[layer->size];
    for (size_t i = 0; i < layer->size; i++) {
        result[i] = layer->neurons[i]->actv_value;
    }
    double *loss_value = nn->loss.derivative(output, result, layer->size);
    // printf("Input : ");
    // print_vector(inputs, nn->layers[0]->input_size);
    // printf("Expected : %f, Predicted : %f, Loss : %f\n", output[0], result[0], loss_value[0]);

    // Put it in the last layer
    for (size_t n = 0; n < layer->size; n++) {
        layer->neurons[n]->error = loss_value[n];
        // transfer derivative is computed later
    }
    free(loss_value);

    // Backpropagate it through every layer
    for (int l = nn->number_of_layers - 1; l > 0; l--) {
        layer = nn->layers[l];
        layer->backprop(layer, nn->layers[l - 1]);
    }
    layer->backprop(nn->layers[0], NULL);

    // Update weights
    for (int l = nn->number_of_layers - 1; l >= 0; l--) {
        struct layer* layer = nn->layers[l];
        for (size_t n = 0; n < layer->size; n++) {
            struct neuron* neuron = layer->neurons[n];
            for (size_t i = 0; i < layer->input_size; i++) {
                double input_value;
                if (l == 0)
                    input_value = inputs[i];
                else
                    input_value = nn->layers[l - 1]->neurons[i]->actv_value;

                double update = (gamma * neuron->momentum) +
                                ((1 - gamma) * learning_rate * neuron->error * input_value);
                // printf("Error : %f, weight : %f, input %f, update : %f\n", neuron->error,
                // neuron->weights[i], input_value, update);
                
                if (update != update) { // is NaN
                    printf("Network has diverged : %f.\n", update);
                    exit(0);
                }
                neuron->weights[i] -= update;
                neuron->momentum = update;
            }
            if (nn->use_bias)
                neuron->bias -= (learning_rate * neuron->error);
        }
    }
}

void
fit(struct neural_network* nn,
    size_t data_size,
    double* inputs[],
    double* outputs[],
    size_t epochs,
    size_t batch_size,
    double learning_rate,
    double gamma)
{
    (void)batch_size;
    reset_values(nn);
    reset_errors(nn);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch : %zu\t-\t", epoch);
        double loss = 0;
        for (size_t i = 0; i < data_size; i++) {
            double* expected = outputs[i];
            double* inp = inputs[i];

            double* result = feed_forward(nn, inp);
            loss += nn->loss.evaluate(expected, result, nn->layers[nn->number_of_layers - 1]->size);
            back_propagate(nn, expected, inp, learning_rate, gamma);

            free(result);
        }
        printf("Mean loss : %f\n", loss / (data_size));
    }
}

double
evaluate(struct neural_network* nn,
         size_t data_size,
         double* inputs[],
         double* outputs[],
         struct loss loss,
         int verbose)
{
    if (0 == data_size)
        return 0;

    double total = 0;
    for (size_t i = 0; i < data_size; i++) {
        double* prediction = feed_forward(nn, inputs[i]);
        double loss_value =
          loss.evaluate(outputs[i], prediction, nn->layers[nn->number_of_layers - 1]->size);
        if (verbose >= 3) {
            printf("Real : ");
            print_vector(outputs[i], nn->layers[nn->number_of_layers - 1]->size);
            printf(" - Prediction : ");
            print_vector(prediction, nn->layers[nn->number_of_layers - 1]->size);
            printf("\n");
        }
        free(prediction);
        if (verbose >= 2)
            printf("\nLoss n°%zu : %f\n", i, loss_value);
        total += loss_value;
    }
    total /= data_size;
    if (verbose >= 1)
        printf("Loss on test set : %f\n", total);
    return total;
}