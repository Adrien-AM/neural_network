#include <iostream>

#include "Tensor.hpp"
#include "Test.hpp"

void
testCreate()
{
    start;

    // Initialization by data
    Tensor<double> t1(vector<double>{ 2, 3 });
    assert(t1.shape() == vector<size_t>{ 2 });
    assert(t1.size() == t1.total_size());
    assert(t1.size() == 2);
    assert(t1[0] == 2);
    assert(t1[1] == 3);

    // Initialization by shape
    Tensor<double> t2(vector<size_t>({ 2, 2 }));

    assert(t2.shape() == vector<size_t>({ 2, 2 }));
    assert(t2.size() != t2.total_size());
    assert(t2.size() == 2);
    assert(t2.total_size() == 4);

    // Initialization by nothing
    Tensor<double> t3;
    assert(t3.size() == t3.total_size());
    assert(t3.size() == 0);
    assert(t3.empty());

    // Initialization by copy
    Tensor<double> tmp(vector<size_t>({ 2, 2 }));
    Tensor<double> t4(tmp);

    assert(t4.shape() == vector<size_t>({ 2, 2 }));
    assert(t4.size() != t4.total_size());
    assert(t4.size() == 2);
    assert(t4.total_size() == 4);

    // Initialization by size
    Tensor<double> t5(3);
    assert(t5.shape() == vector<size_t>({ 3 }));

    end;
}

void
testAffectation()
{
    start;

    Tensor<double> tmp(vector<size_t>({ 2, 2 }));
    Tensor<double> t1(vector<size_t>({ 1, 1 }));
    t1 = tmp;

    assert(t1.shape() == vector<size_t>({ 2, 2 }));
    assert(t1.size() != t1.total_size());
    assert(t1.size() == 2);
    assert(t1.total_size() == 4);

    // Shallow copy : should point to the same node
    assert(&(tmp({ 0, 0 })) == &(t1({ 0, 0 })));

    // Affectation from raw data
    Tensor<double> t2(vector<size_t>({ 3 }));
    t2 = { 1.0, 2.0, 3.0 };
    assert(t2.shape() == vector<size_t>{ 3 });
    assert(t2(0) == 1.0);
    assert(t2(1) == 2.0);
    assert(t2(2) == 3.0);

    // Affectation to empty tensor
    Tensor<double> t3(vector<size_t>({ 0 }));
    t3 = { 1.0, 2.0, 3.0 };
    assert(t3.shape() == vector<size_t>{ 3 });
    assert(t3(0) == 1.0);
    assert(t3(1) == 2.0);
    assert(t3(2) == 3.0);

    end;
}

void
testIndexing()
{
    start;

    // Scalar 1D
    Tensor<double> t1(vector<size_t>{ 3 });
    t1(1) = 2.0;
    assert(t1(0) == 0.0);
    assert(t1(1) == 2.0);
    assert(t1(2) == 0.0);

    // Scalar 2D
    Tensor<double> t2(vector<size_t>{ 3, 2 });
    t2({ 1, 1 }) = 2.0;
    assert(t2({ 1, 0 }) == 0.0);
    assert(t2({ 1, 1 }) == 2.0);
    assert(t2[1](1) == 2.0);

    // Sub array
    Tensor<double> t3(vector<size_t>{ 3, 4 });
    Tensor<double> row = t3[0];
    assert(row.shape() == vector<size_t>({ 4 }));
    row(0) = 1.0;
    assert(t3({ 0, 0 }) == 1);
    // Shallow copy
    assert(&(row(0)) == &(t3({ 0, 0 })));

    end;
}

void
testAddDimension()
{
    start;

    Tensor<double> t(vector<size_t>({ 2, 2 }));
    t.add_dimension();
    assert(t.shape() == vector<size_t>({ 1, 2, 2 }));
    assert(t.total_size() == 4);

    end;
}

void
testFlatten()
{
    start;

    Tensor<double> t(vector<size_t>({ 2, 2 }));
    Tensor<double> flat = t.flatten();
    assert(flat.shape() == vector<size_t>({ 4 }));
    assert(flat.total_size() == 4);

    end;
}

void
testTranspose()
{
    start;

    Tensor<double> t(vector<size_t>({ 2, 3 }));
    t[0] = { 1.0, 2.0, 3.0 };
    t[1] = { 4.0, 5.0, 6.0 };

    Tensor<double> u = t.transpose();
    assert(u.shape() == vector<size_t>({ 3, 2 }));
    assert(u({ 0, 0 }) == 1.0);
    assert(u({ 0, 1 }) == 4.0);
    assert(u({ 1, 0 }) == 2.0);
    assert(u({ 1, 1 }) == 5.0);
    assert(u({ 2, 0 }) == 3.0);
    assert(u({ 2, 1 }) == 6.0);

    end;
}

void
testSum()
{
    start;

    // 1D
    Tensor<double> t1(vector<double>({ 2.0, 3.0, 2.0 }));
    Tensor<double> s1 = t1.sum();
    assert(s1.shape() == vector<size_t>({ 1 }));
    assert(close(s1[0], 7.0));

    // 2D
    Tensor<double> t2(vector<size_t>({ 2, 2 }));
    t2[0] = { 1.0, 2.0 };
    t2[1] = { 3.0, 4.0 };
    assert(close(t2.sum(), 10.0));

    end;
}

void
testMatMul()
{
    start;

    // Vectors
    Tensor<double> v1(vector<size_t>({ 1, 2 }));
    Tensor<double> v2(vector<size_t>({ 2, 1 }));
    v1[0] = { 2.0, 3.0 };
    v2[0] = { 3.0 };
    v2[1] = { 2.0 };
    Tensor<double> result = v1.mm(v2);
    assert(result.shape() == vector<size_t>({ 1, 1 }));
    assert(result({ 0, 0 }) == 12);

    // Matrices
    Tensor<double> m1(vector<size_t>({ 3, 2 }));
    Tensor<double> m2(vector<size_t>({ 2, 3 }));
    m1[0] = { 1.0, 2.0 };
    m1[1] = { 3.0, 4.0 };
    m1[2] = { 5.0, 6.0 };
    m2[0] = { 1.0, 2.0, 3.0 };
    m2[1] = { 4.0, 5.0, 6.0 };

    Tensor<double> r1 = m1.mm(m2);
    assert(r1.shape() == vector<size_t>({ 3, 3 }));
    assert(close(r1({ 0, 0 }), 9.0));
    assert(close(r1({ 0, 1 }), 12.0));
    assert(close(r1({ 0, 2 }), 15.0));
    assert(close(r1({ 1, 0 }), 19.0));
    assert(close(r1({ 1, 1 }), 26.0));
    assert(close(r1({ 1, 2 }), 33.0));
    assert(close(r1({ 2, 0 }), 29.0));
    assert(close(r1({ 2, 1 }), 40.0));
    assert(close(r1({ 2, 2 }), 51.0));

    Tensor<double> r2 = m2.mm(m1);
    assert(r2.shape() == vector<size_t>({ 2, 2 }));
    assert(close(r2({ 0, 0 }), 22.0));
    assert(close(r2({ 0, 1 }), 28.0));
    assert(close(r2({ 1, 0 }), 49.0));
    assert(close(r2({ 1, 1 }), 64.0));

    end;
}

int
main(void)
{
    testCreate();
    testAffectation();
    testIndexing();
    testAddDimension();
    testFlatten();
    testTranspose();
    testSum();
    testMatMul();
}
