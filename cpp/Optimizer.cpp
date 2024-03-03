#include "Optimizer.hpp"

double
clipper(double x, double clip)
{
    if (clip == 0)
        return x;
    return copysign(min(abs(x), clip), x);
}

void
SGD::update()
{
    for (size_t i = 0; i < layers.size(); i++) {
        Layer* l = layers[i];
        vector<size_t> shape = l->weights.shape();
        if (shape.size() == 0)
            continue;
        bool use_bias = !l->biases.empty();

        for (auto& w : l->weights) {
            // printf("Grad update : %f\n", w.gradient);
            w->value -= alpha * w->gradient;
        }

        if (use_bias) {
            for (auto& b : l->biases) {
                b->value -= alpha * b->gradient;
            }
        }
        layers[i]->weights.reset_gradients();
        if (use_bias)
            layers[i]->biases.reset_gradients();
    }
}

Adam::Adam(double lr, double beta1, double beta2, double clip)
  : Optimizer(lr, clip)
  , beta1(beta1)
  , beta2(beta2)
  , t(0)
{
}

void
Adam::update()
{
    for (size_t i = 0; i < layers.size(); i++) {
        t++;
        double bias1 = 1 - pow(beta1, t);
        double bias2 = 1 - pow(beta2, t);

        for (size_t j = 0; j < layers[i]->weights.total_size(); j++) {
            double g = clipper(layers[i]->weights.gradient(j), clip);
            updates1[j] = updates1[j] * beta1 + (1 - beta1) * g;
            updates2[j] = updates2[j] * beta2 + (1 - beta2) * g * g;
            double corrected_up1 = updates1[j] / bias1;
            double corrected_up2 = updates2[j] / bias2;

            layers[i]->weights[j] -= alpha * corrected_up1 / (sqrt(corrected_up2) + 1e-8);
        }
    }
}

void
Adam::attach_layers(vector<Layer*> l)
{
    layers = l;
    for (size_t i = 0; i < layers.size(); i++) {
        this->updates1.push_back(layers[i]->weights.shape());
        this->updates2.push_back(layers[i]->weights.shape());
    }
}
