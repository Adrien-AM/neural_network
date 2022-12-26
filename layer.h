#ifndef __LAYER_H__
#define __LAYER_H__

#include <stdlib.h>

struct layer
{
    struct neuron** neurons;
    size_t size;
    size_t input_size;
    double (*activation)(double, int);
    void (*forward)(struct layer *, struct layer *);
    void (*backprop)(struct layer*, struct layer*);
};

// Allocates memory and initializes neurons. Input size must be known
void
instanciate_neurons(struct layer* layer);

struct layer*
dense_layer(size_t number_of_neurons, double (*activation)(double, int));

#endif // __LAYER_H__