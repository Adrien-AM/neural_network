#include <iostream>

#include "Loss.hpp"

#include "Test.hpp"

void testMAE()
{
    start;
    Tensor<double> y_true = vector<double>{2.2};
    Tensor<double> y_pred = vector<double>{1};

    MeanAbsoluteError loss;
    double result = loss.evaluate(y_true, y_pred);
    assert(close(result, 1.2));
    loss.backward();
    assert(close(y_pred.gradient({0}), -1));

    // Doing it again yields the same results after resetting grads
    y_true.reset_gradients();
    y_pred.reset_gradients();
    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 1.2));
    loss.backward();
    assert(close(y_pred.gradient({0}), -1));

    y_true = vector<double>{-1.5};
    y_pred = vector<double>{2};

    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 3.5));
    loss.backward();
    assert(close(y_pred.gradient({0}), 1));

    y_true = vector<double>{1.5, -3};
    y_pred = vector<double>{1, -2};

    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 0.75));
    loss.backward();
    assert(close(y_pred.gradient({0}), -0.5));
    assert(close(y_pred.gradient({1}), 0.5));

    end;
}

void testMSE()
{
    start;
    Tensor<double> y_true = vector<double>{2.5};
    Tensor<double> y_pred = vector<double>{1};

    MeanSquaredError loss;
    double result = loss.evaluate(y_true, y_pred);
    assert(close(result, 2.25));
    loss.backward();
    assert(close(y_pred.gradient({0}), -3));

    // Doing it again yields the same results after resetting grads
    y_true.reset_gradients();
    y_pred.reset_gradients();
    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 2.25));
    loss.backward();
    assert(close(y_pred.gradient({0}), -3));

    y_true = vector<double>{-1.5};
    y_pred = vector<double>{2};

    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 12.25));
    loss.backward();
    assert(close(y_pred.gradient({0}), 7));

    y_true = vector<double>{1.5, -3};
    y_pred = vector<double>{1, -2};

    result = loss.evaluate(y_true, y_pred);
    assert(close(result, 0.625));
    loss.backward();
    assert(close(y_pred.gradient({0}), -0.5));
    assert(close(y_pred.gradient({1}), 1));

    end;
}

void testCrossEntropy()
{
    start;
    Tensor<double> y_true = vector<double>{0, 1, 0}; // One-hot encoded
    Tensor<double> y_pred = vector<double>{0.1, 0.7, 0.2}; // Softmax probabilities

    CategoricalCrossEntropy loss;
    double result = loss.evaluate(y_true, y_pred);
    assert(close(result, -log(0.7))); // Cross entropy loss
    loss.backward();
    assert(close(y_pred.gradient({1}), -1/0.7)); // Gradient for the true class
    assert(close(y_pred.gradient({0}), 0)); // Gradient for the wrong class
    assert(close(y_pred.gradient({2}), 0)); // Gradient for the wrong class

    // Reset gradients
    y_true.reset_gradients();
    y_pred.reset_gradients();

    // Test with another set
    y_true = vector<double>{1, 0, 0}; // One-hot encoded
    y_pred = vector<double>{0.8, 0.1, 0.1}; // Softmax probabilities

    result = loss.evaluate(y_true, y_pred);
    assert(close(result, -log(0.8))); // Cross entropy loss
    loss.backward();
    assert(close(y_pred.gradient({0}), -1/0.8)); // Gradient for the true class
    assert(close(y_pred.gradient({1}), 0)); // Gradient for the wrong class
    assert(close(y_pred.gradient({2}), 0)); // Gradient for the wrong class

    end;
}


int main()
{
    testMAE();
    testMSE();
    testCrossEntropy();
    return 0;
}
