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
//     uniform_real_distribution<double> dis(-5.0, 5.0);
//     normal_distribution<double> dis_weights(0, 0.5);
//     vector<double> inputs(1000);
//     vector<double> outputs(1000);

//     for (size_t i = 0; i < inputs.size(); ++i) {
//         inputs[i] = dis(gen);
//         outputs[i] = f(inputs[i]);
//     }

//     Number<double>* a = new Number<double>(dis_weights(gen));
//     Number<double>* b = new Number<double>(dis_weights(gen));
//     Number<double>* input = new Number<double>(0.0);
//     Number<double>* output = new Number<double>(0.0);

//     Operation<double>* prediction = new Add(new Mul(a, input), b);
//     CompGraph<double> model(new Pow(new Sub(output, prediction), new Number(2.0)));

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
//     Tensor<double> real = vector<size_t>({ 28, 28 });
//     Tensor<double> predicted = vector<size_t>({ 28, 28 });

//     size_t size = predicted.total_size();
//     double* data = predicted.data();
//     Tensor<Number<double>*> inputs = predicted.shape();
//     Number<double>** variables = inputs.data();
//     if (size == 0)
//         return;
//     for (size_t i = 0; i < size; i++) {
//         variables[i] = new Number<double>(data[i]);
//     }

//     Number<double>* k1 = new Number<double>(0.01);
//     Number<double>* k2 = new Number<double>(0.03);

//     size_t height = real.shape()[0];
//     size_t width = real.shape()[1];

//     // Compute means and variances
//     Operation<double>* mean1 = new Number<double>(0.0);
//     Operation<double>* mean2 = new Number<double>(0.0);

//     Operation<double>* var1 = new Number<double>(0.0);
//     Operation<double>* var2 = new Number<double>(0.0);
//     Operation<double>* cov = new Number<double>(0.0);
//     for (size_t i = 0; i < height; i++) {
//         for (size_t j = 0; j < width; j++) {
//             mean1 = new Add<double>(mean1, new Number<double>(real.at(i)[j]));
//             mean2 = new Add<double>(mean2, inputs.at(i)[j]);
//         }
//     }
//     Number<double>* s = new Number<double>(height * width);
//     mean1 = new Div<double>(mean1, s);
//     mean2 = new Div<double>(mean2, s);

//     for (size_t j = 0; j < width; j++) {
//         for (size_t i = 0; i < height; i++) {
//             Operation<double>* diff1 = new Sub<double>(new Number<double>(real.at(i)[j]), mean1);
//             Operation<double>* diff2 = new Sub<double>(inputs.at(i)[j], mean2);
//             var1 = new Add<double>(var1, new Mul<double>(diff1, diff1));
//             var2 = new Add<double>(var2, new Mul<double>(diff2, diff2));
//             cov = new Add<double>(cov, new Mul<double>(diff1, diff2));
//         }

//         Number<double>* n = new Number<double>(height * width - 1);
//         var1 = new Div<double>(var1, n);
//         var2 = new Div<double>(var2, n);
//         cov = new Div<double>(cov, n);
//     }

//     // Compute SSIM
//     Operation<double>* ssim = new Number<double>(0.0);
//     Operation<double>* numerator = new Mul<double>(
//       new Add<double>(new Mul<double>(new Number<double>(2.0), new Mul<double>(mean1, mean2)),
//                       k1),
//       new Add<double>(new Mul<double>(new Number<double>(2.0), cov), k2));

//     Operation<double>* denominator =
//       new Mul<double>(new Add<double>(new Mul<double>(mean1, mean1),
//                                       new Add<double>(new Mul<double>(mean2, mean2), k1)),
//                       new Add<double>(new Add<double>(var1, var2), k2));
//     ssim = new Div<double>(numerator, denominator);
//     // - to make it a Cost function
//     ssim = new Sub<double>(new Number<double>(0.0), ssim);

//     CompGraph<double>* graph = new CompGraph<double>(ssim);
//     cout << graph->forward() << endl;
//     graph->backward();
//     delete graph;
// }

void
ml()
{
    Number<double>* x1 = new Number<double>(-2.0);
    Number<double>* x2 = new Number<double>(4.0);
    Number<double>* y = new Number<double>(-1.0);
    Number<double>* w1 = new Number<double>(3.0);
    Number<double>* w2 = new Number<double>(2.0);
    Number<double>* b = new Number<double>(3.0);
    Number<double>* lambda = new Number<double>(1.0);

    Operation<double>* dot = new Add<double>(new Mul<double>(w1, x1), new Mul<double>(w2, x2));
    Operation<double>* loss = new Max<double>(new Number<double>(0), new Sub<double>(new Number<double>(1.0), new Mul<double>(y, new Add<double>(dot, b))));
    Operation<double>* norm = new Mul<double>(
      lambda, new Pow<double>(new Add<double>(new Abs<double>(w1), new Abs<double>(w2)), new Number<double>(2)));

    CompGraph<double> graph(new Add<double>(loss, norm));
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

    // Number<double>* x = new Number<double>(3.0);
    // Number<double>* y = new Number<double>(2.0);

    // Operation<double>* z = new Mul<double>(x, x);
    // z = new Add<double>(z, y);
    // CompGraph<double> graph(z);

    // printf("Forward : %f\n", graph.forward());
    // graph.backward();
    // printf("Gradients : %f %f\n", x->gradient, y->gradient);


    return 0;
}
