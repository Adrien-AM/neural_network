#include "Optimizer.hpp"

double
clipper(double x, double clip)
{
    if (clip == 0)
        return x;
    return copysign(min(abs(x), clip), x);
}

void
Optimizer::sample()
{
    for (Layer*& l : layers) {
        l->weights.accumulate_gradients();
        if (!l->biases.empty()) {
            l->biases.accumulate_gradients();
        }
    }
}

void
iterate(Tensor<double>& t)
{
    vector<size_t> s = t.shape();
    size_t nb_dims = s.size();
    vector<size_t> indices(nb_dims);
    while (true) {
        // Code ...

        indices[nb_dims - 1]++;
        size_t i = nb_dims - 1;
        while (indices[i] == s[i]) {
            if (i == 0)
                return;
            indices[i] = 0;
            indices[--i]++;
        }
    }
}

void
SGD::update(size_t batch_size)
{
    for (size_t i = 0; i < layers.size(); i++) {
        Layer* l = layers[i];
        vector<size_t> shape = l->weights.shape();
        if (shape.size() == 0)
            continue;
        bool use_bias = !l->biases.empty();

        size_t nb_dims = shape.size();
        vector<size_t> indices(nb_dims);
        while (true) {
            l->weights(indices) -= alpha * clipper(l->weights.acc(indices) / batch_size, clip);

            size_t i = nb_dims - 1;
            indices[i]++;
            while (indices[i] == shape[i]) {
                if (i == 0)
                    goto bias;
                indices[i] = 0;
                indices[--i]++;
            }
        }
    bias:
        if (use_bias) {
            shape = l->biases.shape();
            size_t nb_dims = shape.size();
            vector<size_t> indices(nb_dims);
            while (true) {
                l->biases(indices) -=
                  alpha * clipper(l->biases.acc(indices) / batch_size, clip);

                size_t i = nb_dims - 1;
                indices[i]++;
                while (indices[i] == shape[i]) {
                    if (i == 0)
                        goto end;
                    indices[i] = 0;
                    indices[--i]++;
                }
            }
        }
    end:
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
Adam::update(size_t batch_size)
{
    for (size_t i = 0; i < layers.size(); i++) {
        t++;
        double bias1 = 1 - pow(beta1, t);
        double bias2 = 1 - pow(beta2, t);

        for (size_t j = 0; j < layers[i]->weights.total_size(); j++) {
            double g = clipper(layers[i]->weights.acc(j) / batch_size, clip);
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
