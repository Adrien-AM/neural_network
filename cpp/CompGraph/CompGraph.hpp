#ifndef __COMP_GRAPH__
#define __COMP_GRAPH__

#include "Operation.hpp"
#include <queue>
#include <vector>

using namespace std;

template<typename T>
class CompGraph
{
  private:
    vector<SmartPointer<Operation<double>>> nodes;

  public:
    SmartPointer<Operation<double>> root;

    CompGraph(SmartPointer<Operation<double>> o)
      : root(o)
    {
        queue<SmartPointer<Operation<double>>> q;
        q.push(root);
        while (!q.empty()) {
            SmartPointer<Operation<double>> op = q.front();
            nodes.push_back(op);
            for (auto& input : op->inputs) {
                q.push(input);
            }
            q.pop();
        }
    }

    void backward()
    {
        // this->reset();
        root->gradient = 1;
        for (auto& n : nodes) {
            n->backward();
        }
    }

    // void reset()
    // {
    //     for (auto& n : nodes) {
    //         n->gradient = 0;
    //     }
    // }

    ~CompGraph() {}
};

#endif // __COMP_GRAPH__