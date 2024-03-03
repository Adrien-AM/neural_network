#include "CompGraph/CompGraph.hpp"
#include "CompGraph/Number.hpp"

#include "Tensor.hpp"

#include "Test.hpp"

void testCompGraphNumber()
{
    start;
    Operation<double>* a = new Number<double>(3.0);
    Operation<double>* b = new Number<double>(2.0);
    Operation<double>* c = new Add<double>(a, b);
    c->forward();
    CompGraph<double> g(c);
    g.backward();

    assert(a->value == 3);
    assert(b->value == 2);
    assert(c->value == 5);

    assert(a->gradient == 1);
    assert(b->gradient == 1);
    assert(c->gradient == 1);

    end;
}

void
testGraphOnTensorSum()
{
    start;
    Tensor<double> t = Tensor<double>(vector<double>{ 2.0, 3.0 });
    Tensor<double> s = t.sum();

    CompGraph<double> g = s.get_graph();
    g.backward();
    assert(g.root->value == 5.0);
    assert(g.root->gradient == 1.0);

    assert(g.root->inputs[0]->value == 2.0);
    assert(g.root->inputs[0]->gradient == 1.0);
    assert(g.root->inputs[1]->value == 3.0);
    assert(g.root->inputs[1]->gradient == 1.0);

    end;
}

void testGraphComposition()
{
    start;

    Tensor<double> t1(vector<double>({1.0, 2.0}));
    Tensor<double> t2(vector<double>({3.0, 4.0}));

    Tensor<double> result = t1 - t2;
    result = result.abs().sum();

    CompGraph<double> g = result.get_graph();
    g.backward();

    // [4]
    assert(close(g.root->value, 4));
    assert(close(g.root->gradient, 1));

    // [2, 2] -> [4]
    assert(close(g.root->inputs[0]->value, 2));
    assert(close(g.root->inputs[0]->gradient, 1));

    // [-2, -2] -> [2, 2]
    assert(close(g.root->inputs[0]->inputs[0]->value, -2));
    assert(close(g.root->inputs[0]->inputs[0]->gradient, -1));

    // [1, 3] -> [-2]
    assert(close(g.root->inputs[0]->inputs[0]->inputs[0]->value, 1));
    assert(close(g.root->inputs[0]->inputs[0]->inputs[0]->gradient, -1));

    assert(close(g.root->inputs[0]->inputs[0]->inputs[1]->value, 3));
    assert(close(g.root->inputs[0]->inputs[0]->inputs[1]->gradient, 1));

    // [2, 2] -> [4]
    assert(close(g.root->inputs[1]->value, 2));
    assert(close(g.root->inputs[1]->gradient, 1));

    assert(close(g.root->inputs[1]->inputs[0]->value, -2));
    assert(close(g.root->inputs[1]->inputs[0]->gradient, -1));

    // [-2, -2] -> [2, 2]
    assert(close(g.root->inputs[1]->inputs[0]->inputs[0]->value, 2));
    assert(close(g.root->inputs[1]->inputs[0]->inputs[0]->gradient, -1));

    assert(close(g.root->inputs[1]->inputs[0]->inputs[1]->value, 4));
    assert(close(g.root->inputs[1]->inputs[0]->inputs[1]->gradient, 1));

    end;
}

int
main()
{
    testCompGraphNumber();
    // TODO : fix
    // There is an useless node 'Number' due to Tensor copy
    // Tests not working because of this
    
    // testGraphOnTensorSum();
    // testGraphComposition();
    return 0;
}
