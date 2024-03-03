#include "Layer.hpp"

void Layer::summarize() const
{
    printf("Default layer (probably a missing description)\n");
}

void Layer::print_layer() const
{
    printf("Default print for layer. Maybe you should implement a print function for the layer "
           "you're debugging :)\n");
}

Layer::~Layer()
{}