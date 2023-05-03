#include "Optimizer.hpp"

double
clipper(double x, double clip)
{
    if (clip == 0)
        return x;
    return copysign(min(abs(x), clip), x);
}

void
SGD::update(vector<Tensor<double>> gradients)
{
    for (size_t i = 0; i < layers.size(); i++) {
        double* grads = gradients[i].data();
        double* weights = layers[i]->weights.data();

        for (size_t j = 0; j < gradients[i].total_size(); j++) {
            weights[j] -= clipper(grads[j], clip) * alpha;
        }
    }
    alpha *= decay;
}

Adam::Adam(double lr, double beta1, double beta2, double clip)
  : Optimizer(lr, clip)
  , beta1(beta1)
  , beta2(beta2)
  , t(0)
{
}

void
Adam::update(vector<Tensor<double>> gradients)
{
    for (size_t i = 0; i < layers.size(); i++) {
        double* grads = gradients[i].data();
        double* up1 = updates1[i].data();
        double* up2 = updates2[i].data();
        double* weights = layers[i]->weights.data();

        t++;
        double bias1 = 1 - pow(beta1, t);
        double bias2 = 1 - pow(beta2, t);

        for (size_t j = 0; j < gradients[i].total_size(); j++) {
            double g = clipper(grads[j], clip);
            up1[j] = beta1 * up1[j] + (1 - beta1) * g;
            up2[j] = beta2 * up2[j] + (1 - beta2) * g * g;
            double corrected_up1 = up1[j] / bias1;
            double corrected_up2 = up2[j] / bias2;

            weights[j] -= alpha * corrected_up1 / (sqrt(corrected_up2) + 1e-8);
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