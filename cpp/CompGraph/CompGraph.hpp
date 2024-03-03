#ifndef __COMP_GRAPH__
#define __COMP_GRAPH__

#include "Operation.hpp"
#include <queue>
#include <vector>
#include <unordered_map>

using namespace std;

template<typename T>
class CompGraph
{
  private:
    vector<SmartPointer<Operation<T>>> nodes;

  public:
    SmartPointer<Operation<T>> root;

    CompGraph(SmartPointer<Operation<T>> o)
      : root(o)
    {
        queue<SmartPointer<Operation<T>>> q;
        q.push(root);
        while (!q.empty()) {
            SmartPointer<Operation<T>> op = q.front();
            nodes.push_back(op);
            for (auto& input : op->inputs) {
                q.push(input);
            }
            q.pop();
        }

        // Now keep only the last occurence of each node for it to be topologically sorted
        unordered_map<SmartPointer<Operation<T>>*, int> nodeIndex;
        nodeIndex.reserve(nodes.size());
        for (int i = nodes.size() - 1; i >= 0; --i) {
            if (nodeIndex.find(&nodes[i]) == nodeIndex.end()) {
                nodeIndex[&nodes[i]] = i;
            }
        }
        vector<SmartPointer<Operation<T>>> uniqueNodes;
        uniqueNodes.reserve(nodeIndex.size());
        for (const auto& pair : nodeIndex) {
            uniqueNodes.push_back(*(pair.first));
        }
        nodes = move(uniqueNodes);
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