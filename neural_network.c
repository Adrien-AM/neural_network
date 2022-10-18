#include "neural_network.h"
#include "utils.h"

struct neuron
{
    float *weights;
    float bias;
    float value;
    float actv_value;
    float error;
    float momentum;
};

struct layer
{
    struct neuron **neurons;
    size_t size;
    size_t input_size;
    float (*activation)(float, int);
};

struct neural_network
{
    struct layer **layers;
    size_t number_of_layers;
};

struct neural_network *create_model(size_t number_of_layers, size_t layers_size[],
                                    size_t input_size, float (**activation)(float, int))
{
    struct neural_network *nn = (struct neural_network *)malloc(sizeof(struct neural_network));
    nn->number_of_layers = number_of_layers;
    nn->layers = (struct layer **)malloc(sizeof(struct layer *) * nn->number_of_layers);
    for (size_t l = 0; l < number_of_layers; l++)
    {
        // instanciate layer
        struct layer *layer = (struct layer *)malloc(sizeof(struct layer));
        layer->size = layers_size[l];
        layer->activation = activation[l];
        layer->input_size = l == 0 ? input_size : nn->layers[l - 1]->size;

        // instanciate array of neurons
        layer->neurons = (struct neuron **)malloc(sizeof(struct neuron *) * layer->size);
        for (size_t n = 0; n < layer->size; n++)
        {
            struct neuron *neuron = (struct neuron *)malloc(sizeof(struct neuron));

            // First layer receives input
            neuron->weights = (float *)calloc(layer->input_size, sizeof(float));
            neuron->bias = 0;
            neuron->value = 0;
            neuron->actv_value = 0;
            neuron->error = 0;
            neuron->momentum = 0;

            layer->neurons[n] = neuron;
        }

        nn->layers[l] = layer;
    }

    return nn;
}

void free_neural_network(struct neural_network *nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++)
    {
        struct layer *layer = nn->layers[l];
        for (size_t n = 0; n < layer->size; n++)
        {
            free(layer->neurons[n]->weights);
            free(layer->neurons[n]);
        }
        free(layer->neurons);
        free(layer);
    }
    free(nn->layers);
    free(nn);
}

void reset_values(struct neural_network *nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++)
    {
        for (size_t n = 0; n < nn->layers[l]->size; n++)
        {
            struct neuron *neuron = nn->layers[l]->neurons[n];
            neuron->value = neuron->bias;
            neuron->actv_value = 0;
        }
    }
}

void reset_errors(struct neural_network *nn)
{
    for (size_t l = 0; l < nn->number_of_layers; l++)
    {
        for (size_t n = 0; n < nn->layers[l]->size; n++)
        {
            struct neuron *neuron = nn->layers[l]->neurons[n];
            neuron->error = 0;
            neuron->momentum = 0;
        }
    }
}

void randomize_weights(struct neural_network *nn, float mu, float sigma, int use_bias)
{
    for (size_t l = 0; l < nn->number_of_layers; l++)
    {
        for (size_t n = 0; n < nn->layers[l]->size; n++)
        {
            for (size_t i = 0; i < nn->layers[l]->input_size; i++)
            {
                nn->layers[l]->neurons[n]->weights[i] = rand_normal(mu, sigma);
            }
            if(use_bias)
                nn->layers[l]->neurons[n]->bias = rand_normal(mu, sigma);
        }
    }
}

float *feed_forward(struct neural_network *nn, float inputs[])
{
    reset_values(nn);
    struct layer *layer = nn->layers[0];
    for (size_t i = 0; i < layer->input_size; i++)
    {
        for (size_t n = 0; n < layer->size; n++)
        {
            layer->neurons[n]->value += inputs[i] * layer->neurons[n]->weights[i]; // xi * wi
        }
    }

    for (size_t l = 1; l < nn->number_of_layers; l++)
    {
        layer = nn->layers[l];
        for (size_t i = 0; i < layer->input_size; i++)
        {
            struct layer *input_layer = nn->layers[l - 1];
            input_layer->neurons[i]->actv_value = input_layer->activation(input_layer->neurons[i]->value, 0);
            for (size_t n = 0; n < layer->size; n++)
            {
                layer->neurons[n]->value += input_layer->neurons[i]->actv_value * layer->neurons[n]->weights[i]; // xi * wi
            }
        }
    }

    struct layer *last_layer = nn->layers[nn->number_of_layers - 1];
    float *result = (float *)malloc(sizeof(float) * last_layer->size); // last layer values
    for (size_t n = 0; n < last_layer->size; n++)
    {
        last_layer->neurons[n]->actv_value = last_layer->activation(last_layer->neurons[n]->value, 0);
        result[n] = last_layer->neurons[n]->actv_value;
    }

    return result;
}

void back_propagate(struct neural_network *nn, float *output, float inputs[], float learning_rate, float gamma)
{
    reset_errors(nn);
    // Output layer
    struct layer *layer = nn->layers[nn->number_of_layers - 1];
    for (size_t n = 0; n < layer->size; n++)
    {
        struct neuron *neuron = layer->neurons[n];
        neuron->error = (output[n] - neuron->actv_value); // error = (value - output) * transfer_derivative(value)
        // transfer derivative is computed later
    }

    for (int l = nn->number_of_layers - 1; l >= 0; l--)
    {
        layer = nn->layers[l];
        for (size_t n = 0; n < layer->size; n++)
        {
            struct neuron *neuron = layer->neurons[n];

            // here error = dactv
            // compute derivative on current neuron error
            neuron->error *= layer->activation(neuron->actv_value, 1);
            // here error = dz

            // propagate to previous layer, weighted
            for (size_t i = 0; i < layer->input_size; i++)
            {
                if (l != 0) // First layer doesn't have to backpropagate
                    nn->layers[l - 1]->neurons[i]->error += neuron->error * neuron->weights[i];
            }
        }
    }

    // Update weights
    for (int l = nn->number_of_layers - 1; l >= 0; l--)
    {
        struct layer *layer = nn->layers[l];
        for (size_t n = 0; n < layer->size; n++)
        {
            struct neuron *neuron = layer->neurons[n];
            for (size_t i = 0; i < layer->input_size; i++)
            {
                float input_value;
                if (l == 0)
                {
                    input_value = inputs[i];
                }
                else
                {
                    input_value = nn->layers[l - 1]->neurons[i]->actv_value;
                }

                float update = (gamma * neuron->momentum) + ((1 - gamma) * learning_rate * neuron->error * input_value);
                neuron->weights[i] += update;
                neuron->momentum = update;
            }
            neuron->bias += (learning_rate * neuron->error);
        }
    }
}

void fit(struct neural_network *nn, size_t data_size, float *inputs[], float *outputs[], size_t epochs, float learning_rate, float gamma)
{
    reset_values(nn);
    reset_errors(nn);

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch : %zu\n", epoch);
        float loss = 0;
        for (size_t i = 0; i < data_size; i++)
        {
            float *expected = outputs[i];
            float *inp = inputs[i];

            float *result = feed_forward(nn, inp);

            for (size_t v = 0; v < nn->layers[0]->input_size; v++) {
                loss += fabsf(result[v] - expected[v]);
            }

            back_propagate(nn, expected, inp, learning_rate, gamma);

            free(result);
        }
        printf("Mean loss : %f\n", loss / (data_size * nn->layers[nn->number_of_layers - 1]->size));
    }
}
