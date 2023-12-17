#include "Tensor.hpp"
#include "Test.hpp"

void
testAdd()
{
    start;

    // Scalar
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> t2 = t1 + 3.0;
    assert(t2.shape() == t1.shape());
    assert(close(t2(0), 4.0));
    assert(close(t2(1), 5.0));
    t2 += 1.0;
    assert(close(t2(0), 5.0));
    assert(close(t2(1), 6.0));

    // Tensor
    // 1D
    Tensor<double> t3(vector<double>{ 1.0, 2.0 });
    Tensor<double> t4(vector<double>{ 3.0, 4.0 });
    Tensor<double> result = t3 + t4;
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), 4.0));
    assert(close(result(1), 6.0));
    t3 += t4;
    assert(close(t3(0), 4.0));
    assert(close(t3(1), 6.0));
    assert(close(result(0), 4.0));
    assert(close(result(1), 6.0));

    // 2D
    Tensor<double> t5(vector<size_t>{ 2, 2 });
    Tensor<double> t6(vector<size_t>{ 2, 2 });
    t5[0] = { 1.0, 2.0 };
    t5[1] = { 3.0, 4.0 };
    t6[0] = { 5.0, 6.0 };
    t6[1] = { 7.0, 8.0 };
    result = t5 + t6;
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), 6.0));
    assert(close(result({ 0, 1 }), 8.0));
    assert(close(result({ 1, 0 }), 10.0));
    assert(close(result({ 1, 1 }), 12.0));
    t5 += t6;
    assert(close(t5({ 0, 0 }), 6.0));
    assert(close(t5({ 0, 1 }), 8.0));
    assert(close(t5({ 1, 0 }), 10.0));
    assert(close(t5({ 1, 1 }), 12.0));

    end;
}

void
testSub()
{
    start;

    // Scalar
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> t2 = t1 - 2.0;
    assert(t2.shape() == t1.shape());
    assert(close(t2(0), -1.0));
    assert(close(t2(1), 0.0));
    t2 -= 1.0;
    assert(close(t2(0), -2.0));
    assert(close(t2(1), -1.0));

    // Tensor
    // 1D
    Tensor<double> t3(vector<double>{ 1.0, 2.0 });
    Tensor<double> t4(vector<double>{ 3.0, 4.0 });
    Tensor<double> result = t3 - t4;
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), -2.0));
    assert(close(result(1), -2.0));
    t3 -= t4;
    assert(close(t3(0), -2.0));
    assert(close(t3(1), -2.0));
    assert(close(result(0), -2.0));
    assert(close(result(1), -2.0));

    // 2D
    Tensor<double> t5(vector<size_t>{ 2, 2 });
    Tensor<double> t6(vector<size_t>{ 2, 2 });
    t5[0] = { 1.0, 2.0 };
    t5[1] = { 3.0, 4.0 };
    t6[0] = { 5.0, 6.0 };
    t6[1] = { 7.0, 8.0 };
    result = t5 - t6;
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), -4.0));
    assert(close(result({ 0, 1 }), -4.0));
    assert(close(result({ 1, 0 }), -4.0));
    assert(close(result({ 1, 1 }), -4.0));
    t5 -= t6;
    assert(close(t5({ 0, 0 }), -4.0));
    assert(close(t5({ 0, 1 }), -4.0));
    assert(close(t5({ 1, 0 }), -4.0));
    assert(close(t5({ 1, 1 }), -4.0));

    end;
}

void
testMul()
{
    start;

    // Scalar
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> t2 = t1 * 2.0;
    assert(t2.shape() == t1.shape());
    assert(close(t2(0), 2.0));
    assert(close(t2(1), 4.0));
    t2 *= 3.0;
    assert(close(t2(0), 6.0));
    assert(close(t2(1), 12.0));

    // Tensor
    // 1D
    Tensor<double> t3(vector<double>{ 1.0, 2.0 });
    Tensor<double> t4(vector<double>{ 3.0, 4.0 });
    Tensor<double> result = t3 * t4;
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), 3.0));
    assert(close(result(1), 8.0));
    t3 *= t4;
    assert(close(t3(0), 3.0));
    assert(close(t3(1), 8.0));
    assert(close(result(0), 3.0));
    assert(close(result(1), 8.0));

    // 2D
    Tensor<double> t5(vector<size_t>{ 2, 2 });
    Tensor<double> t6(vector<size_t>{ 2, 2 });
    t5[0] = { 1.0, 2.0 };
    t5[1] = { 3.0, 4.0 };
    t6[0] = { 5.0, 6.0 };
    t6[1] = { 7.0, 8.0 };
    result = t5 * t6;
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), 5.0));
    assert(close(result({ 0, 1 }), 12.0));
    assert(close(result({ 1, 0 }), 21.0));
    assert(close(result({ 1, 1 }), 32.0));
    t5 *= t6;
    assert(close(t5({ 0, 0 }), 5.0));
    assert(close(t5({ 0, 1 }), 12.0));
    assert(close(t5({ 1, 0 }), 21.0));
    assert(close(t5({ 1, 1 }), 32.0));

    end;
}

void
testDiv()
{
    start;

    // Scalar
    Tensor<double> t1(vector<double>{ 2.0, 4.0 });
    Tensor<double> t2 = t1 / 2.0;
    assert(t2.shape() == t1.shape());
    assert(close(t2(0), 1.0));
    assert(close(t2(1), 2.0));
    t2 /= 2.0;
    assert(close(t2(0), 0.5));
    assert(close(t2(1), 1.0));

    // Tensor
    // 1D
    Tensor<double> t3(vector<double>{ 6.0, 8.0 });
    Tensor<double> t4(vector<double>{ 2.0, 4.0 });
    Tensor<double> result = t3 / t4;
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), 3.0));
    assert(close(result(1), 2.0));
    t3 /= t4;
    assert(close(t3(0), 3.0));
    assert(close(t3(1), 2.0));
    assert(close(result(0), 3.0));
    assert(close(result(1), 2.0));

    // 2D
    Tensor<double> t5(vector<size_t>{ 2, 2 });
    Tensor<double> t6(vector<size_t>{ 2, 2 });
    t5[0] = { 10.0, 12.0 };
    t5[1] = { 14.0, 16.0 };
    t6[0] = { 2.0, 3.0 };
    t6[1] = { 4.0, 5.0 };
    result = t5 / t6;
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), 5.0));
    assert(close(result({ 0, 1 }), 4.0));
    assert(close(result({ 1, 0 }), 3.5));
    assert(close(result({ 1, 1 }), 3.2));
    t5 /= t6;
    assert(close(t5({ 0, 0 }), 5.0));
    assert(close(t5({ 0, 1 }), 4.0));
    assert(close(t5({ 1, 0 }), 3.5));
    assert(close(t5({ 1, 1 }), 3.2));

    end;
}

void
testExp()
{
    start;

    // Tensor
    // 1D
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> result = t1.exp();
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), exp(1.0)));
    assert(close(result(1), exp(2.0)));

    // 2D
    Tensor<double> t2(vector<size_t>{ 2, 2 });
    t2[0] = { 0.0, 1.0 };
    t2[1] = { 2.0, 3.0 };
    result = t2.exp();
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), exp(0.0)));
    assert(close(result({ 0, 1 }), exp(1.0)));
    assert(close(result({ 1, 0 }), exp(2.0)));
    assert(close(result({ 1, 1 }), exp(3.0)));

    end;
}

void
testLog()
{
    start;

    // Tensor
    // 1D
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> result = t1.log();
    assert(result.shape() == vector<size_t>({ 2 }));
    assert(close(result(0), log(1.0)));
    assert(close(result(1), log(2.0)));

    // 2D
    Tensor<double> t2(vector<size_t>{ 2, 2 });
    t2[0] = { 1.0, 2.0 };
    t2[1] = { 3.0, 4.0 };
    result = t2.log();
    assert(result.shape() == vector<size_t>({ 2, 2 }));
    assert(close(result({ 0, 0 }), log(1.0)));
    assert(close(result({ 0, 1 }), log(2.0)));
    assert(close(result({ 1, 0 }), log(3.0)));
    assert(close(result({ 1, 1 }), log(4.0)));

    end;
}

void
testMax()
{
    start;

    // Tensor/Scalar
    Tensor<double> t1(vector<double>{ 1.0, 2.0 });
    Tensor<double> max1 = t1.max(1.5);
    assert(max1.shape() == t1.shape());
    assert(close(max1(0), 1.5)); // Max of 1.0 and 1.5 is 1.5
    assert(close(max1(1), 2.0)); // Max of 2.0 and 1.5 is 2.0

    // Tensor/Tensor
    Tensor<double> t2(vector<double>{ 2.0, 3.0 });
    Tensor<double> max2 = t1.max(t2);
    assert(max2.shape() == t1.shape());
    assert(close(max2(0), 2.0)); // Max of 1.0 and 2.0 is 2.0
    assert(close(max2(1), 3.0)); // Max of 2.0 and 3.0 is 3.0

    end;
}

void
testPow()
{
    start;

    // Tensor/Scalar
    Tensor<double> t1(vector<double>{ 2.0, 3.0 });
    Tensor<double> pow1 = t1.pow(2.0);
    assert(pow1.shape() == t1.shape());
    assert(close(pow1(0), 4.0)); // 2.0 ^ 2.0 is 4.0
    assert(close(pow1(1), 9.0)); // 3.0 ^ 2.0 is 9.0

    // Tensor/Tensor
    Tensor<double> t2(vector<double>{ 2.0, 3.0 });
    Tensor<double> pow2 = t1.pow(t2);
    assert(pow2.shape() == t1.shape());
    assert(close(pow2(0), 4.0));  // 2.0 ^ 2.0 is 4.0
    assert(close(pow2(1), 27.0)); // 3.0 ^ 3.0 is 27.0

    end;
}

void
testSigm()
{
    start;

    // Tensor
    // 1D
    Tensor<double> t1(vector<double>{ 0.0, 1.0, -1.0 });
    Tensor<double> result = t1.sigm();
    assert(result.shape() == vector<size_t>({ 3 }));

    // Sigmoid function for 0.0 is 0.5
    assert(close(result(0), 0.5));
    // Sigmoid function for 1.0 is approximately 0.73106
    assert(close(result(1), 0.73106));
    // Sigmoid function for -1.0 is approximately 0.26894
    assert(close(result(2), 0.26894));

    // 2D
    Tensor<double> t2(vector<size_t>{ 2, 2 });
    t2[0] = { 0.0, 2.0 };
    t2[1] = { -1.0, -3.0 };
    result = t2.sigm();
    assert(result.shape() == vector<size_t>({ 2, 2 }));

    // Sigmoid function for 0.0 is 0.5
    assert(close(result({ 0, 0 }), 0.5));
    // Sigmoid function for 2.0 is approximately 0.88079
    assert(close(result({ 0, 1 }), 0.88079));
    // Sigmoid function for -1.0 is approximately 0.26894
    assert(close(result({ 1, 0 }), 0.26894));
    // Sigmoid function for -3.0 is approximately 0.04743
    assert(close(result({ 1, 1 }), 0.04743));

    end;
}

void
testAbs()
{
    start;

    // Tensor
    // 1D
    Tensor<double> t1(vector<double>{ -1.0, 2.0, -3.5 });
    Tensor<double> result = t1.abs();
    assert(result.shape() == vector<size_t>({ 3 }));

    // Absolute value of -1.0 is 1.0
    assert(close(result(0), 1.0));
    // Absolute value of 2.0 is 2.0
    assert(close(result(1), 2.0));
    // Absolute value of -3.5 is 3.5
    assert(close(result(2), 3.5));

    // 2D
    Tensor<double> t2(vector<size_t>{ 2, 2 });
    t2[0] = { -1.0, 2.0 };
    t2[1] = { -3.5, 0.0 };
    result = t2.abs();
    assert(result.shape() == vector<size_t>({ 2, 2 }));

    // Absolute value of -1.0 is 1.0
    assert(close(result({ 0, 0 }), 1.0));
    // Absolute value of 2.0 is 2.0
    assert(close(result({ 0, 1 }), 2.0));
    // Absolute value of -3.5 is 3.5
    assert(close(result({ 1, 0 }), 3.5));
    // Absolute value of 0.0 is 0.0
    assert(close(result({ 1, 1 }), 0.0));

    end;
}

void
testComposition()
{
    start;

    Tensor<double> t1(vector<size_t>({ 2, 2 }));
    Tensor<double> t2(vector<size_t>({ 2, 2 }));
    t1[0] = { 1.0, 2.0 };
    t1[1] = { 3.0, 4.0 };
    t2[0] = { 4.0, 3.0 };
    t2[1] = { 2.0, 1.0 };

    Tensor<double> t3 = (t2 - t1).pow(2.0);
    assert(t3.shape() == vector<size_t>({ 2, 2 }));
    assert(close(t3({ 0, 0 }), 9.0));
    assert(close(t3({ 0, 1 }), 1.0));
    assert(close(t3({ 1, 0 }), 1.0));
    assert(close(t3({ 1, 1 }), 9.0));

    end;
}

void
testMatMul()
{
    start;

    Tensor<double> t1(vector<size_t>({ 2, 1 }));
    t1({ 0, 0 }) = 2.0;
    t1({ 1, 0 }) = 3.0;
    Tensor<double> t2(vector<size_t>({ 1, 2 }));
    t2[0] = { 1.0, 0.0 };

    Tensor<double> t3 = t2.mm(t1);
    CompGraph<double> g = t3.get_graph();
    g.backward();

    SmartPointer<Operation<double>> o = g.root;
    assert(close(g.root->value, 2.0));
    assert(close(g.root->gradient, 1.0));

    o = o->inputs[0];
    assert(close(o->value, 2.0));
    assert(close(o->gradient, 1.0));

    end;
}

int
main()
{
    testAdd();
    testSub();
    testMul();
    testDiv();
    testExp();
    testLog();
    testMax();
    testPow();
    testSigm();
    testAbs();

    testComposition();
    testMatMul();
    return 0;
}
