#include <iostream>

#include "../Tensor.hpp"
#include "Abs.hpp"
#include "Add.hpp"
#include "CompGraph.hpp"
#include "Div.hpp"
#include "Exp.hpp"
#include "Max.hpp"
#include "Mul.hpp"
#include "Pow.hpp"
#include "Sigm.hpp"
#include "Sub.hpp"
#include "Number.hpp"

#include <random>

double
f(double x)
{
    return 0.3 * x + 1.2;
}

// void
// regression()
// {
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution<T> dis(-5.0, 5.0);
//     normal_distribution<T> dis_weights(0, 0.5);
//     vector<T> inputs(1000);
//     vector<T> outputs(1000);

//     for (size_t i = 0; i < inputs.size(); ++i) {
//         inputs[i] = dis(gen);
//         outputs[i] = f(inputs[i]);
//     }

//     Number<T>* a = new Number<T>(dis_weights(gen));
//     Number<T>* b = new Number<T>(dis_weights(gen));
//     Number<T>* input = new Number<T>(0.0);
//     Number<T>* output = new Number<T>(0.0);

//     Operation<T>* prediction = new Add(new Mul(a, input), b);
//     CompGraph<T> model(new Pow(new Sub(output, prediction), new Number(2.0)));

//     double learning_rate = 0.01;
//     for (size_t i = 0; i < inputs.size(); i++) {
//         printf("Input : %f ; Output : %f ; ", inputs[i], outputs[i]);
//         input->value = inputs[i];
//         output->value = outputs[i];
//         double result = model.forward();
//         printf("Prediction : %f, Loss : %f\n", prediction->value, result);
//         model.backward();
//         printf("Gradients : %f %f\n", a->gradient, b->gradient);
//         a->value -= learning_rate * a->gradient;
//         b->value -= learning_rate * b->gradient;
//     }

//     printf("----\nFinal : y = %fx + %f\n----\n", a->value, b->value);
// }

// void
// test()
// {
//     Tensor<T> real = vector<size_t>({ 28, 28 });
//     Tensor<T> predicted = vector<size_t>({ 28, 28 });

//     size_t size = predicted.total_size();
//     double* data = predicted.data();
//     Tensor<Number<T>*> inputs = predicted.shape();
//     Number<T>** variables = inputs.data();
//     if (size == 0)
//         return;
//     for (size_t i = 0; i < size; i++) {
//         variables[i] = new Number<T>(data[i]);
//     }

//     Number<T>* k1 = new Number<T>(0.01);
//     Number<T>* k2 = new Number<T>(0.03);

//     size_t height = real.shape()[0];
//     size_t width = real.shape()[1];

//     // Compute means and variances
//     Operation<T>* mean1 = new Number<T>(0.0);
//     Operation<T>* mean2 = new Number<T>(0.0);

//     Operation<T>* var1 = new Number<T>(0.0);
//     Operation<T>* var2 = new Number<T>(0.0);
//     Operation<T>* cov = new Number<T>(0.0);
//     for (size_t i = 0; i < height; i++) {
//         for (size_t j = 0; j < width; j++) {
//             mean1 = new Add<T>(mean1, new Number<T>(real.at(i)[j]));
//             mean2 = new Add<T>(mean2, inputs.at(i)[j]);
//         }
//     }
//     Number<T>* s = new Number<T>(height * width);
//     mean1 = new Div<T>(mean1, s);
//     mean2 = new Div<T>(mean2, s);

//     for (size_t j = 0; j < width; j++) {
//         for (size_t i = 0; i < height; i++) {
//             Operation<T>* diff1 = new Sub<T>(new Number<T>(real.at(i)[j]), mean1);
//             Operation<T>* diff2 = new Sub<T>(inputs.at(i)[j], mean2);
//             var1 = new Add<T>(var1, new Mul<T>(diff1, diff1));
//             var2 = new Add<T>(var2, new Mul<T>(diff2, diff2));
//             cov = new Add<T>(cov, new Mul<T>(diff1, diff2));
//         }

//         Number<T>* n = new Number<T>(height * width - 1);
//         var1 = new Div<T>(var1, n);
//         var2 = new Div<T>(var2, n);
//         cov = new Div<T>(cov, n);
//     }

//     // Compute SSIM
//     Operation<T>* ssim = new Number<T>(0.0);
//     Operation<T>* numerator = new Mul<T>(
//       new Add<T>(new Mul<T>(new Number<T>(2.0), new Mul<T>(mean1, mean2)),
//                       k1),
//       new Add<T>(new Mul<T>(new Number<T>(2.0), cov), k2));

//     Operation<T>* denominator =
//       new Mul<T>(new Add<T>(new Mul<T>(mean1, mean1),
//                                       new Add<T>(new Mul<T>(mean2, mean2), k1)),
//                       new Add<T>(new Add<T>(var1, var2), k2));
//     ssim = new Div<T>(numerator, denominator);
//     // - to make it a Cost function
//     ssim = new Sub<T>(new Number<T>(0.0), ssim);

//     CompGraph<T>* graph = new CompGraph<T>(ssim);
//     cout << graph->forward() << endl;
//     graph->backward();
//     delete graph;
// }

void
ml()
{
    Number<T>* x1 = new Number<T>(-2.0);
    Number<T>* x2 = new Number<T>(4.0);
    Number<T>* y = new Number<T>(-1.0);
    Number<T>* w1 = new Number<T>(3.0);
    Number<T>* w2 = new Number<T>(2.0);
    Number<T>* b = new Number<T>(3.0);
    Number<T>* lambda = new Number<T>(1.0);

    Operation<T>* dot = new Add<T>(new Mul<T>(w1, x1), new Mul<T>(w2, x2));
    Operation<T>* loss = new Max<T>(new Number<T>(0), new Sub<T>(new Number<T>(1.0), new Mul<T>(y, new Add<T>(dot, b))));
    Operation<T>* norm = new Mul<T>(
      lambda, new Pow<T>(new Add<T>(new Abs<T>(w1), new Abs<T>(w2)), new Number<T>(2)));

    CompGraph<T> graph(new Add<T>(loss, norm));
    cout << graph.forward() << endl;
    graph.backward();

    printf("Gradients : w1=%f, w2=%f, b=%f\n", w1->gradient, w2->gradient, b->gradient);
}

int
main()
{
    // regression();
    // test();
    ml();

    // Number<T>* x = new Number<T>(3.0);
    // Number<T>* y = new Number<T>(2.0);

    // Operation<T>* z = new Mul<T>(x, x);
    // z = new Add<T>(z, y);
    // CompGraph<T> graph(z);

    // printf("Forward : %f\n", graph.forward());
    // graph.backward();
    // printf("Gradients : %f %f\n", x->gradient, y->gradient);


    return 0;
}
