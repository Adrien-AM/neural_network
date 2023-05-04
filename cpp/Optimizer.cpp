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
    size_t size = layers.size();
    for (size_t i = 0; i < size; i++) {
        double* grads = gradients[i].data();
        double* up1 = updates1[i].data();
        double* up2 = updates2[i].data();
        double* weights = layers[i]->weights.data();

        t++;
        double bias1 = 1 - pow(beta1, t);
        double bias2 = 1 - pow(beta2, t);

        double grads_size = gradients[i].total_size();
        for (size_t j = 0; j < grads_size; j++) {
            double g = clipper(grads[j], clip);
            double& r1 = up1[j];
            double& r2 = up2[j];
            r1 = beta1 * r1 + (1 - beta1) * g;
            r2 = beta2 * r2 + (1 - beta2) * g * g;
            double corrected_up1 = r1 / bias1;
            double corrected_up2 = r2 / bias2;

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
