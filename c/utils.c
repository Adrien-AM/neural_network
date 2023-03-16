#include "utils.h"

void
print_vector(double* v, size_t size)
{
    printf("[ ");
    for (size_t i = 0; i < size - 1; i++) {
        printf("%f ", v[i]);
    }
    printf("%f ]", v[size - 1]);
}

void
print_softmax(double* result, size_t nb_classes, double until)
{
    double max = 0;
    size_t imax = 0;
    do {
        for (size_t i = 0; i < nb_classes; i++) {
            if (result[i] > max) {
                imax = i;
                max = result[i];
            }
        }
        printf("Class %zu : %f%%\t-\t", imax, max * 100);
        result[imax] = 0;
        max = 0;
    } while (max > until);
    printf("Others too low.\n");
}

double
relu(double x, int derivative)
{
    if (derivative)
        return x >= 0 ? 1 : 0;
    return x >= 0 ? x : 0;
}

double
sigmoid(double x, int derivative)
{
    if (derivative) {
        double out = sigmoid(x, 0);
        return out * (1 - out);
    }

    return 1 / (1 + exp(-x));
}

double
linear(double x, int derivative)
{
    if (derivative)
        return 1;
    return x;
}

double
hypertan(double x, int derivative)
{
    if (derivative) {
        double r = tanh(x);
        return 1 - (r * r);
    }
    return tanh(x);
}

// CREDITS
// https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus

double
rand_gen()
{
    return ((double)rand() + 1) / ((double)RAND_MAX + 1);
}

double
rand_normal(double mu, double sigma)
{
    double v1 = rand_gen();
    double v2 = rand_gen();

    double r1 = cos(2 * 3.14 * v2) * sqrt(-2 * log(v1));
    return r1 * sigma + mu;
}

// Mean Squared Error evaluation
double
mse_f(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        if (error > 1e12) {
            printf("pred %f, true %f\n", y_pred[i], y_true[i]);
            exit(0);
        }
        total += (error * error - total) / (i + 1); // avoid overflow :)
    }
    // printf("err total : %f\n", total);

    return total;
}

// Mean Squared Error derivative
double*
mse_d(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;

    double* result = malloc(sizeof(double) * size);
    for (size_t i = 0; i < size; i++) {
        result[i] = 2 * (y_pred[i] - y_true[i]);
    }

    return result;
}

// Mean Absolute Error evaluation
double
mae_f(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;
    double total = 0;
    for (size_t i = 0; i < size; i++) {
        double error = y_pred[i] - y_true[i];
        total += fabs(error);
    }
    return total / size;
}

// Mean Absolute Error derivative
double*
mae_d(double* y_true, double* y_pred, size_t size)
{
    if (0 == size)
        return 0;

    double* result = malloc(sizeof(double) * size);

    for (size_t i = 0; i < size; i++) {
        result[i] = y_pred[i] - y_true[i];
    }

    return result;
}

// Cross Entropy evaluation
double
ce_f(double* y_true, double* y_pred, size_t size)
{
    double loss = 0;
    for (size_t i = 0; i < size; i++) {
        loss -= (y_true[i] * log(y_pred[i]));
        // printf("Adding %f\n", (y_true[i] * log(y_pred[i])));
    }

    return loss;
}

// Cross Entropy derivative
double*
ce_d(double* y_true, double* y_pred, size_t size)
{
    double* result = malloc(sizeof(double) * size);

    for (size_t i = 0; i < size; i++) {
        result[i] = -(y_true[i] / y_pred[i]);
    }

    return result;
}

double
psnr_f(double* original, double* reconstructed, size_t size)
{
    double mse = mse_f(original, reconstructed, size);
    return 10 * log10(255 * 255 / mse);
}

double*
psnr_d(double* original, double* reconstructed, size_t size)
{
    double* result = malloc(sizeof(double) * size);
    for (size_t i = 0; i < size; i++) {
        result[i] = -10 / (log(10) * (reconstructed[i] - original[i]));
    }
    return result;
}

const double C1 = 6.5025, C2 = 58.5225;
// Structural Similarity
// By ChatGPT
double
ssim_window(double* img1, double* img2, size_t width, size_t height)
{
    double mean_x = 0, mean_y = 0, var_x = 0, var_y = 0, cov_xy = 0;

    // calculate mean, variance and covariance
    for (size_t i = 0; i < width * height; i++) {
        mean_x += img1[i];
        mean_y += img2[i];
    }
    mean_x /= width * height;
    mean_y /= width * height;
    for (size_t i = 0; i < width * height; i++) {
        var_x += (img1[i] - mean_x) * (img1[i] - mean_x);
        var_y += (img2[i] - mean_y) * (img2[i] - mean_y);
        cov_xy += (img1[i] - mean_x) * (img2[i] - mean_y);
    }
    var_x /= width * height - 1;
    var_y /= width * height - 1;
    cov_xy /= width * height - 1;

    // calculate SSIM
    double ssim = (2 * mean_x * mean_y + C1) * (2 * cov_xy + C2) /
                  ((mean_x * mean_x + mean_y * mean_y + C1) * (var_x + var_y + C2));
    return -ssim;
}

double
ssim_f(double* img1, double* img2, size_t size)
{
    size_t width, height;
    width = height = sqrt(size);

    const size_t window_size = 14;
    double total = 0;
    size_t nb_windows = 0;
    for (size_t i = 0; i < width; i += window_size) {
        for (size_t j = 0; j < height; j += window_size) {
            total +=
              ssim_window((img1 + j * width + i), (img2 + j * width + i), window_size, window_size);
            nb_windows++;
        }
    }
    return total / nb_windows;
}

const double delta = 1e-2;

double*
ssim_d(double* img1, double* img2, size_t size)
{
    double* partials = malloc(sizeof(double) * size);
    for (size_t i = 0; i < size; i++) {
        img2[i] += delta;
        double f_x_plus_delta = ssim_f(img1, img2, size);
        img2[i] -= delta;
        double f_x = ssim_f(img1, img2, size);
        partials[i] = (f_x_plus_delta - f_x) / delta;
        // printf("%f\n", partials[i]);
    }
    return partials;
}

const struct loss mean_squared_error = { &mse_f, &mse_d };
const struct loss mean_absolute_error = { &mae_f, &mae_d };
const struct loss cross_entropy = { &ce_f, &ce_d };
const struct loss psnr = { &psnr_f, &psnr_d };
const struct loss ssim = { &ssim_f, &ssim_d };