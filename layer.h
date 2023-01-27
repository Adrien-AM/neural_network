#ifndef __LAYER_H__
#define __LAYER_H__

#include <stdlib.h>

/*
 * @var neurons : array of neurons
 * @var size : number of neurons
 * @var input_size : number of inputs
 * @var activation : (x, derivative) : f if derivative is false, f' if true
 * @var forward : (layer, input_layer) : computes the actv_value for each neuron of this layer,
 * taking the actv_value of input_layer as inputs
 * @var backprop : (layer, input_layer) : propagate the error of this layer to the precedent layer.
 * Supposes the error of actv_value is already known for each neuron. input_layer may be NULL if
 * this is the first layer.
 */
struct layer
{
    struct neuron** neurons;
    size_t size;
    size_t input_size;
    double (*activation)(double, int);
    void (*forward)(struct layer*, struct layer*);
    void (*backprop)(struct layer*, struct layer*);
};

// Allocates memory and initializes neurons. Input size must be known
void
instanciate_neurons(struct layer* layer);

/*
* Instanciates a simple, fully connected dense layer of size `number_of_neurons.
* @param number_of_neurons
* @param activation
*/
struct layer*
dense_layer(size_t number_of_neurons, double (*activation)(double, int));

/*
* Instanciates a softmax layer.
* The previous layer should have number_of_classes nodes.
* The activation of previous layer should be linear or ReLU for better results.
* @param number_of_classes
*/
struct layer*
softmax_layer(size_t number_of_classes);


/*
* Instanciates a dropout layer. Drop rate is the same for every dropout layer, creating a new one will override old drop rate.
*/
struct layer*
dropout_layer(size_t size, double drop_rate);

#endif // __LAYER_H__