#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include "Tensor.hpp"
#include <string>
#include <vector>

using namespace std;

template<typename T>
class Dataset
{
  public:
    Dataset() {}
    virtual vector<Tensor<T>> get_item(size_t index, size_t nb) = 0;
    virtual size_t size() const = 0;
};

class MnistData : public Dataset<double>
{
  private:
    // Constants
    size_t W = 28;
    size_t H = 28;
    
    string path_img;
    string path_labels;
    size_t nb_images;
    size_t count_mnist_images(string path);

  public:
    MnistData(string path, string path_labels, size_t nb);
    vector<Tensor<double>> get_item(size_t index, size_t nb);
    size_t size() const;
};

#endif // __DATASET_HPP__