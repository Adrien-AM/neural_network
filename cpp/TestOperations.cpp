#include <iostream>

#include "CompGraph/Abs.hpp"
#include "CompGraph/Add.hpp"
#include "CompGraph/Div.hpp"
#include "CompGraph/Exp.hpp"
#include "CompGraph/Log.hpp"
#include "CompGraph/Max.hpp"
#include "CompGraph/Mul.hpp"
#include "CompGraph/Pow.hpp"
#include "CompGraph/Sigm.hpp"
#include "CompGraph/Sub.hpp"
#include "CompGraph/Number.hpp"

#include "Test.hpp"

void
testAdd()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(3);
    SmartPointer<Operation<double>> v2 = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Add<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(5 == o->value);
    assert(2 == o->gradient);

    assert(3 == v1->value);
    assert(2 == v2->value);
    assert(2 == v1->gradient);
    assert(2 == v2->gradient);

    end;
}

void
testSub()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(3);
    SmartPointer<Operation<double>> v2 = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Sub<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(1 == o->value);
    assert(2 == o->gradient);

    assert(3 == v1->value);
    assert(2 == v2->value);
    assert(2 == v1->gradient);
    assert(-2 == v2->gradient);

    end;
}

void
testMul()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(3);
    SmartPointer<Operation<double>> v2 = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Mul<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(6 == o->value);
    assert(2 == o->gradient);

    assert(3 == v1->value);
    assert(2 == v2->value);
    assert(4 == v1->gradient);
    assert(6 == v2->gradient);

    end;
}

void
testDiv()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(3);
    SmartPointer<Operation<double>> v2 = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Div<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(1.5 == o->value);
    assert(2 == o->gradient);

    assert(3 == v1->value);
    assert(2 == v2->value);
    assert(1 == v1->gradient); // Approximate value
    assert(-1.5 == v2->gradient);

    end;
}

void
testAbs()
{
    start;
    SmartPointer<Operation<double>> v = new Number<double>(-3);
    SmartPointer<Operation<double>> o = new Abs<double>(v);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(3, o->value));
    assert(close(2, o->gradient));

    assert(close(-3, v->value));
    assert(close(-2, v->gradient));

    v = new Number<double>(3);
    o = new Abs<double>(v);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(3, o->value));
    assert(close(2, o->gradient));

    assert(close(3, v->value));
    assert(close(2, v->gradient));

    end;
}

void
testExp()
{
    start;
    SmartPointer<Operation<double>> v = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Exp<double>(v);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(std::exp(2), o->value));
    assert(close(2, o->gradient));

    assert(close(2, v->value));
    assert(close(2 * std::exp(2), v->gradient));

    end;
}

void
testLog()
{
    start;
    SmartPointer<Operation<double>> v = new Number<double>(4);
    SmartPointer<Operation<double>> o = new Log<double>(v);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(std::log(4), o->value));
    assert(close(2, o->gradient));

    assert(close(4, v->value));
    assert(close(2.0 / 4.0, v->gradient));

    end;
}

void
testMax()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(3);
    SmartPointer<Operation<double>> v2 = new Number<double>(2);
    SmartPointer<Operation<double>> o = new Max<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(3, o->value));
    assert(close(2, o->gradient));

    assert(close(3, v1->value));
    assert(close(2, v1->gradient));

    assert(close(2, v2->value));
    assert(close(0, v2->gradient));

    end;
}

void
testPow()
{
    start;
    SmartPointer<Operation<double>> v1 = new Number<double>(2);
    SmartPointer<Operation<double>> v2 = new Number<double>(3);
    SmartPointer<Operation<double>> o = new Pow<double>(v1, v2);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(8, o->value));
    assert(close(24, v1->gradient));              // Exact gradient value
    assert(close(8 * std::log(2), v2->gradient)); // Exact gradient value

    end;
}

void
testSigm()
{
    start;
    SmartPointer<Operation<double>> v = new Number<double>(1);
    SmartPointer<Operation<double>> o = new Sigm<double>(v);

    o->forward();
    o->gradient = 2;
    o->backward();

    assert(close(1 / (1 + std::exp(-1)), o->value)); // Exact value

    assert(close(1, v->value));
    assert(close(2 * (1 / (1 + std::exp(-1))) * (1 - (1 / (1 + std::exp(-1)))),
                 v->gradient)); // Exact gradient value

    end;
}

int
main()
{
    testAdd();
    testSub();
    testMul();
    testDiv();
    testAbs();
    testExp();
    testLog();
    testMax();
    testPow();
    testSigm();
    return 0;
}
