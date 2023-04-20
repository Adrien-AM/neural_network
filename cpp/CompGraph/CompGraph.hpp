#ifndef __COMP_GRAPH__
#define __COMP_GRAPH__

#include "Operation.hpp"
#include <vector>
#include <set>

using namespace std;

template<typename T>
class CompGraph
{
  private:
    void create_nodes(Operation<T>* o)
    {
        for (const auto& input : o->inputs) {
            create_nodes(input);
        }
        nodes.push_back(o);
    }

  public:
    vector<Operation<T>*> nodes;

    CompGraph()
      : nodes(0)
    {
    }

    CompGraph(Operation<T>* o)
    {
        nodes = vector<Operation<T>*>(0);
        create_nodes(o);
    }

    T forward()
    {
        for (const auto& o : nodes)
            o->forward();
        return nodes.back()->value;
    }

    void backward()
    {
        this->reset();
        nodes.back()->gradient = 1;
        set<Operation<T>*> uniques;
        for (int i = nodes.size() - 1; i >= 0; i--) {
            if(!uniques.count(nodes[i])) {
                nodes[i]->backward();
                uniques.insert(nodes[i]);
            }
        }
    }

    void reset()
    {
        for (const auto& o : nodes)
            o->gradient = 0;
    }

    ~CompGraph()
    {
        // Remove dupes
        set<Operation<T>*> uniques(nodes.cbegin(), nodes.cend());
        for (auto& o : uniques) {
            delete o;
        }
    }
};

#endif // __COMP_GRAPH__